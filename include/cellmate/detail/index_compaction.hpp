#pragma once

#include <cstddef>
#include <cstdint>

#include "simd.hpp"

namespace cellmate
{
	namespace detail
	{
		constexpr bool has_index_compaction = DistQuery::N_lanes > 1;

#if CELLMATE_DETAIL_SIMD_WIDTH == 256
		// builds the lane permutation that packs entries selected by the mask to the front
		// (shared by float and index packing)
		inline __m256i contiguous_lane_shuffle(unsigned mask) noexcept
		{
			// move the selected lanes to the front in order, one source lane per 4 bits
			// (the first selected lane is in the lowest 4 bits, decoded below)
			static constexpr std::uint32_t shuffle_table[256]{
				0x00000000, 0x00000000, 0x00000001, 0x00000010, 0x00000002, 0x00000020, 0x00000021, 0x00000210,
				0x00000003, 0x00000030, 0x00000031, 0x00000310, 0x00000032, 0x00000320, 0x00000321, 0x00003210,
				0x00000004, 0x00000040, 0x00000041, 0x00000410, 0x00000042, 0x00000420, 0x00000421, 0x00004210,
				0x00000043, 0x00000430, 0x00000431, 0x00004310, 0x00000432, 0x00004320, 0x00004321, 0x00043210,
				0x00000005, 0x00000050, 0x00000051, 0x00000510, 0x00000052, 0x00000520, 0x00000521, 0x00005210,
				0x00000053, 0x00000530, 0x00000531, 0x00005310, 0x00000532, 0x00005320, 0x00005321, 0x00053210,
				0x00000054, 0x00000540, 0x00000541, 0x00005410, 0x00000542, 0x00005420, 0x00005421, 0x00054210,
				0x00000543, 0x00005430, 0x00005431, 0x00054310, 0x00005432, 0x00054320, 0x00054321, 0x00543210,
				0x00000006, 0x00000060, 0x00000061, 0x00000610, 0x00000062, 0x00000620, 0x00000621, 0x00006210,
				0x00000063, 0x00000630, 0x00000631, 0x00006310, 0x00000632, 0x00006320, 0x00006321, 0x00063210,
				0x00000064, 0x00000640, 0x00000641, 0x00006410, 0x00000642, 0x00006420, 0x00006421, 0x00064210,
				0x00000643, 0x00006430, 0x00006431, 0x00064310, 0x00006432, 0x00064320, 0x00064321, 0x00643210,
				0x00000065, 0x00000650, 0x00000651, 0x00006510, 0x00000652, 0x00006520, 0x00006521, 0x00065210,
				0x00000653, 0x00006530, 0x00006531, 0x00065310, 0x00006532, 0x00065320, 0x00065321, 0x00653210,
				0x00000654, 0x00006540, 0x00006541, 0x00065410, 0x00006542, 0x00065420, 0x00065421, 0x00654210,
				0x00006543, 0x00065430, 0x00065431, 0x00654310, 0x00065432, 0x00654320, 0x00654321, 0x06543210,
				0x00000007, 0x00000070, 0x00000071, 0x00000710, 0x00000072, 0x00000720, 0x00000721, 0x00007210,
				0x00000073, 0x00000730, 0x00000731, 0x00007310, 0x00000732, 0x00007320, 0x00007321, 0x00073210,
				0x00000074, 0x00000740, 0x00000741, 0x00007410, 0x00000742, 0x00007420, 0x00007421, 0x00074210,
				0x00000743, 0x00007430, 0x00007431, 0x00074310, 0x00007432, 0x00074320, 0x00074321, 0x00743210,
				0x00000075, 0x00000750, 0x00000751, 0x00007510, 0x00000752, 0x00007520, 0x00007521, 0x00075210,
				0x00000753, 0x00007530, 0x00007531, 0x00075310, 0x00007532, 0x00075320, 0x00075321, 0x00753210,
				0x00000754, 0x00007540, 0x00007541, 0x00075410, 0x00007542, 0x00075420, 0x00075421, 0x00754210,
				0x00007543, 0x00075430, 0x00075431, 0x00754310, 0x00075432, 0x00754320, 0x00754321, 0x07543210,
				0x00000076, 0x00000760, 0x00000761, 0x00007610, 0x00000762, 0x00007620, 0x00007621, 0x00076210,
				0x00000763, 0x00007630, 0x00007631, 0x00076310, 0x00007632, 0x00076320, 0x00076321, 0x00763210,
				0x00000764, 0x00007640, 0x00007641, 0x00076410, 0x00007642, 0x00076420, 0x00076421, 0x00764210,
				0x00007643, 0x00076430, 0x00076431, 0x00764310, 0x00076432, 0x00764320, 0x00764321, 0x07643210,
				0x00000765, 0x00007650, 0x00007651, 0x00076510, 0x00007652, 0x00076520, 0x00076521, 0x00765210,
				0x00007653, 0x00076530, 0x00076531, 0x00765310, 0x00076532, 0x00765320, 0x00765321, 0x07653210,
				0x00007654, 0x00076540, 0x00076541, 0x00765410, 0x00076542, 0x00765420, 0x00765421, 0x07654210,
				0x00076543, 0x00765430, 0x00765431, 0x07654310, 0x00765432, 0x07654320, 0x07654321, 0x76543210
			};

			// expand the packed source lanes into one permutation index per output lane
			// (the permutation only reads the low 3 bits of each index)
			return _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(shuffle_table[mask & 0xffu])),
				_mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28));
		}
#endif

		// packs floats of a contiguous array selected by the mask into 'packed'
		// (requires 'DistQuery::N_lanes' input entries and output slots)
		inline void pack_contiguous_floats(const float* values, unsigned mask, float* packed) noexcept
		{
#if CELLMATE_DETAIL_SIMD_WIDTH == 512
			static_assert(DistQuery::N_lanes == 16, "AVX512 float compaction requires 16 lanes");
			_mm512_storeu_ps(packed, _mm512_maskz_compress_ps(static_cast<__mmask16>(mask), _mm512_loadu_ps(values)));

#elif CELLMATE_DETAIL_SIMD_WIDTH == 256
			static_assert(DistQuery::N_lanes == 8, "AVX2 float compaction requires 8 lanes");
			_mm256_storeu_ps(packed, _mm256_permutevar8x32_ps(_mm256_loadu_ps(values), contiguous_lane_shuffle(mask)));
#else
			// collect matching values one at a time
			while (mask != 0)
			{
				*packed++ = values[first_set_bit(mask)];
				mask &= mask - 1;
			}
#endif
		}

		// packs entries of a contiguous index array selected by the mask into 'neighbors' and returns their count
		// (requires 'DistQuery::N_lanes' input entries and output slots)
		inline std::size_t pack_contiguous_indices(const std::uint32_t* indices, unsigned mask, std::uint32_t* neighbors) noexcept
		{
#if CELLMATE_DETAIL_SIMD_WIDTH == 512
			static_assert(DistQuery::N_lanes == 16, "AVX512 index compaction requires 16 lanes");
			_mm512_storeu_si512(neighbors, _mm512_maskz_compress_epi32(static_cast<__mmask16>(mask), _mm512_loadu_si512(indices)));
#if defined(_MSC_VER)
			return __popcnt(mask);
#else
			return static_cast<std::size_t>(__builtin_popcount(mask));
#endif

#elif CELLMATE_DETAIL_SIMD_WIDTH == 256
			static_assert(DistQuery::N_lanes == 8, "AVX2 index compaction requires 8 lanes");
			const __m256i packed = _mm256_permutevar8x32_epi32(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices)), contiguous_lane_shuffle(mask));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(neighbors), packed);
#if defined(_MSC_VER)
			return __popcnt(mask & 0xffu);
#else
			return static_cast<std::size_t>(__builtin_popcount(mask & 0xffu));
#endif
#else
			// collect matching indices one at a time
			std::size_t N_indices = 0;
			while (mask != 0)
			{
				neighbors[N_indices++] = indices[first_set_bit(mask)];
				mask &= mask - 1;
			}
			return N_indices;
#endif
		}
	}
}
