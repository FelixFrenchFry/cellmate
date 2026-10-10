#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__AVX2__) || defined(CELLMATE_SIMD_AVX512)
#include <immintrin.h>
#endif

#include "simd.hpp"

namespace cellmate
{
	namespace detail
	{

#if (defined(__AVX2__) || defined(CELLMATE_SIMD_AVX512)) && SIZE_MAX == UINT64_MAX
		constexpr bool has_index_compaction = DistQuery::N_lanes > 1;
#else
		constexpr bool has_index_compaction = false;
#endif

		// pack indices selected by the mask into neighbors and return their count
		// (requires N_lanes input entries and output slots)
		template<class PointEntry>
		inline std::size_t pack_neighbor_indices(const PointEntry* point_order, unsigned mask, std::size_t* neighbors) noexcept
		{
			if (mask == 0) { return 0; }
			std::size_t N_indices = 0;

#if defined(CELLMATE_SIMD_AVX512) && SIZE_MAX == UINT64_MAX
			// load original indices in point order, then store those selected by the mask
			std::size_t point_indices[8];
			for (std::size_t lane = 0; lane < 8; lane++)
			{
				point_indices[lane] = point_order[lane].idx;
			}
			const __m512i indices = _mm512_loadu_si512(point_indices);
			_mm512_mask_compressstoreu_epi64(neighbors, static_cast<__mmask8>(mask), indices);

			while (mask != 0)
			{
				N_indices++;
				mask &= mask - 1;
			}
			return N_indices;

#elif defined(__AVX2__) && SIZE_MAX == UINT64_MAX
			if (DistQuery::N_lanes == 4)
			{
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
			}
#endif

			// collect matching indices one at a time
			while (mask != 0)
			{
				neighbors[N_indices++] = point_order[first_set_bit(mask)].idx;
				mask &= mask - 1;
			}
			return N_indices;
		}
	}
}
