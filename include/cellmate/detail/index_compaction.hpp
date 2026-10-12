#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "simd.hpp"

namespace cellmate
{
	namespace detail
	{
		constexpr bool has_index_compaction = DistQuery::N_lanes > 1 && SIZE_MAX == UINT64_MAX;

		// transposes neighbor match masks from one mask per candidate to one mask per query
		// (writes candidate selection masks to 'query_masks', with one bit per candidate)
		template<std::size_t N_lanes>
		inline void transpose_neighbor_masks(const unsigned* candidate_masks, unsigned* query_masks) noexcept
		{
			static_assert(N_lanes == 1 || N_lanes == 8 || N_lanes == 16, "unsupported query mask width");
			const std::size_t N_block_lanes = N_lanes < 8 ? N_lanes : 8;
			for (std::size_t lane = 0; lane < N_lanes; lane++)
			{
				query_masks[lane] = 0;
			}

			// transpose 1 block of 8x8 bits for 8-lane version, or 4 blocks of 8x8 bits for 16-lane version
			// TODO: extract each query's candidate mask directly in case of AVX512?
			for (std::size_t first_query = 0; first_query < N_lanes; first_query += 8)
			{
				for (std::size_t first_candidate = 0; first_candidate < N_lanes; first_candidate += 8)
				{
					// store candidate masks as rows of an 8x8 bit block
					uint64_t bits = 0;
					for (std::size_t candidate = 0; candidate < N_block_lanes; candidate++)
					{
						bits |= static_cast<uint64_t>((candidate_masks[first_candidate + candidate] >> first_query) & 0xffu) << (candidate * 8);
					}

					// transpose the bit block by swapping bits across its diagonal
					uint64_t swapped = (bits ^ (bits >> 7)) & UINT64_C(0x00aa00aa00aa00aa);
					bits ^= swapped ^ (swapped << 7);
					swapped = (bits ^ (bits >> 14)) & UINT64_C(0x0000cccc0000cccc);
					bits ^= swapped ^ (swapped << 14);
					swapped = (bits ^ (bits >> 28)) & UINT64_C(0x00000000f0f0f0f0);
					bits ^= swapped ^ (swapped << 28);
					
					// merge each transposed row into its query's candidate mask
					for (std::size_t query = 0; query < N_block_lanes; query++)
					{
						query_masks[first_query + query] |= static_cast<unsigned>((bits >> (query * 8)) & 0xffu) << first_candidate;
					}
				}
			}
		}

		// pack indices selected by the mask into 'neighbors' and return their count
		// (requires 'DistQuery::N_lanes' input entries and output slots)
		template<class PointEntry>
		inline std::size_t pack_neighbor_indices(const PointEntry* point_order, unsigned mask, std::size_t* neighbors) noexcept
		{
			if (mask == 0) { return 0; }

#if CELLMATE_DETAIL_SIMD_WIDTH == 512 && SIZE_MAX == UINT64_MAX
			static_assert(DistQuery::N_lanes == 16, "AVX512 index compaction requires 16 lanes");
			static_assert(std::is_standard_layout<PointEntry>::value &&
				sizeof(PointEntry) == 16 && offsetof(PointEntry, idx) == 8,
				"AVX512 index loads require '{ uint64_t key; size_t idx; }' entries");

			const unsigned first_mask = mask & 0xffu;
			const unsigned second_mask = (mask >> 8) & 0xffu;
#if defined(_MSC_VER)
			const std::size_t N_first = __popcnt(first_mask);
			const std::size_t N_second = __popcnt(second_mask);
#else
			const std::size_t N_first = static_cast<std::size_t>(__builtin_popcount(first_mask));
			const std::size_t N_second = static_cast<std::size_t>(__builtin_popcount(second_mask));
#endif

			// extract indices from the odd 64-bit lanes and pack both halves in order
			const __m512i odd_lanes = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
			if (first_mask != 0)
			{
				const __m512i first_entries = _mm512_loadu_si512(point_order);
				const __m512i second_entries = _mm512_loadu_si512(point_order + 4);
				const __m512i point_indices = _mm512_permutex2var_epi64(first_entries, odd_lanes, second_entries);
				_mm512_mask_compressstoreu_epi64(neighbors, static_cast<__mmask8>(first_mask), point_indices);
			}
			if (second_mask != 0)
			{
				const __m512i first_entries = _mm512_loadu_si512(point_order + 8);
				const __m512i second_entries = _mm512_loadu_si512(point_order + 12);
				const __m512i point_indices = _mm512_permutex2var_epi64(first_entries, odd_lanes, second_entries);
				_mm512_mask_compressstoreu_epi64(neighbors + N_first, static_cast<__mmask8>(second_mask), point_indices);
			}
			return N_first + N_second;

#elif CELLMATE_DETAIL_SIMD_WIDTH == 256 && SIZE_MAX == UINT64_MAX
			static_assert(DistQuery::N_lanes == 8, "AVX2 index compaction requires 8 lanes");
			static_assert(std::is_standard_layout<PointEntry>::value &&
				sizeof(PointEntry) == 16 && offsetof(PointEntry, idx) == 8,
				"AVX2 index loads require '{ uint64_t key; size_t idx; }' entries");

			// shuffle both 32-bit halves of each selected index together
			// (preserves 64-bit indices and their original order)
			static constexpr std::uint32_t shuffle_table[16][8]{
				{ 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 1, 0, 0, 0, 0, 0, 0 },
				{ 2, 3, 0, 0, 0, 0, 0, 0 },
				{ 0, 1, 2, 3, 0, 0, 0, 0 },
				{ 4, 5, 0, 0, 0, 0, 0, 0 },
				{ 0, 1, 4, 5, 0, 0, 0, 0 },
				{ 2, 3, 4, 5, 0, 0, 0, 0 },
				{ 0, 1, 2, 3, 4, 5, 0, 0 },
				{ 6, 7, 0, 0, 0, 0, 0, 0 },
				{ 0, 1, 6, 7, 0, 0, 0, 0 },
				{ 2, 3, 6, 7, 0, 0, 0, 0 },
				{ 0, 1, 2, 3, 6, 7, 0, 0 },
				{ 4, 5, 6, 7, 0, 0, 0, 0 },
				{ 0, 1, 4, 5, 6, 7, 0, 0 },
				{ 2, 3, 4, 5, 6, 7, 0, 0 },
				{ 0, 1, 2, 3, 4, 5, 6, 7 }
			};

			static constexpr unsigned N_indices_by_mask[16]{ 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

			// collect indices in point order and pack both halves of the mask
			const unsigned first_mask = mask & 0xfu;
			const unsigned second_mask = (mask >> 4) & 0xfu;
			std::size_t N_indices = 0;
			if (first_mask != 0)
			{
				const __m256i first_entries = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(point_order));
				const __m256i second_entries = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(point_order + 2));
				const __m256i interleaved_indices = _mm256_unpackhi_epi64(first_entries, second_entries);
				const __m256i point_indices = _mm256_permute4x64_epi64(interleaved_indices, 0xd8);
				const __m256i shuffle = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(shuffle_table[first_mask]));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(neighbors), _mm256_permutevar8x32_epi32(point_indices, shuffle));
				N_indices = N_indices_by_mask[first_mask];
			}
			if (second_mask != 0)
			{
				const __m256i first_entries = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(point_order + 4));
				const __m256i second_entries = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(point_order + 6));
				const __m256i interleaved_indices = _mm256_unpackhi_epi64(first_entries, second_entries);
				const __m256i point_indices = _mm256_permute4x64_epi64(interleaved_indices, 0xd8);
				const __m256i shuffle = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(shuffle_table[second_mask]));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(neighbors + N_indices), _mm256_permutevar8x32_epi32(point_indices, shuffle));
				N_indices += N_indices_by_mask[second_mask];
			}
			return N_indices;
#else
			// collect matching indices one at a time
			std::size_t N_indices = 0;
			while (mask != 0)
			{
				neighbors[N_indices++] = point_order[first_set_bit(mask)].idx;
				mask &= mask - 1;
			}
			return N_indices;
#endif
		}
	}
}
