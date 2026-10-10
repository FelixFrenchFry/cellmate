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

		// pack indices selected by the mask into neighbors and return their count
		// (requires N_lanes input entries and output slots)
		template<class PointEntry>
		inline std::size_t pack_neighbor_indices(const PointEntry* point_order, unsigned mask, std::size_t* neighbors) noexcept
		{
			if (mask == 0) { return 0; }

#if CELLMATE_DETAIL_SIMD_WIDTH == 512 && SIZE_MAX == UINT64_MAX
			static_assert(DistQuery::N_lanes == 8, "AVX512 index compaction requires 8 lanes");
			static_assert(std::is_standard_layout<PointEntry>::value &&
				sizeof(PointEntry) == 16 && offsetof(PointEntry, idx) == 8,
				"AVX512 index loads require { uint64_t key; size_t idx; } entries");

			// collect the original indices in point order from the odd 64-bit lanes of both entry groups
			// (avoids scalar stores followed by a wide load, which cannot be store-forwarded)
			const __m512i first_entries = _mm512_loadu_si512(point_order);
			const __m512i second_entries = _mm512_loadu_si512(point_order + 4);
			const __m512i odd_lanes = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
			const __m512i point_indices = _mm512_permutex2var_epi64(first_entries, odd_lanes, second_entries);

			// store the matching indices contiguously
			_mm512_mask_compressstoreu_epi64(neighbors, static_cast<__mmask8>(mask), point_indices);
#if defined(_MSC_VER)
			return __popcnt(mask);
#else
			return static_cast<std::size_t>(__builtin_popcount(mask));
#endif
#elif CELLMATE_DETAIL_SIMD_WIDTH == 256 && SIZE_MAX == UINT64_MAX
			static_assert(DistQuery::N_lanes == 4, "AVX2 index compaction requires 4 lanes");
			static_assert(std::is_standard_layout<PointEntry>::value &&
				sizeof(PointEntry) == 16 && offsetof(PointEntry, idx) == 8,
				"AVX2 index loads require { uint64_t key; size_t idx; } entries");

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

			// collect the original indices in point order
			const __m256i first_entries = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(point_order));
			const __m256i second_entries = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(point_order + 2));
			const __m256i interleaved_indices = _mm256_unpackhi_epi64(first_entries, second_entries);
			const __m256i point_indices = _mm256_permute4x64_epi64(interleaved_indices, 0xd8);

			// move matching indices to the front and store the full group
			const __m256i shuffle = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(shuffle_table[mask]));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(neighbors), _mm256_permutevar8x32_epi32(point_indices, shuffle));
			return N_indices_by_mask[mask];
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
