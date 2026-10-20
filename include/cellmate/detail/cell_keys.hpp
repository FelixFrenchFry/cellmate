#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "fp_control.hpp"
#include "simd.hpp"

// match backend settings so position scans and cell keys can inline into the rebuild and batch loops
CELLMATE_DETAIL_FP_BEGIN

namespace cellmate
{
	namespace detail
	{
#if CELLMATE_DETAIL_SIMD_WIDTH == 512 || CELLMATE_DETAIL_SIMD_WIDTH == 256
		// expands the bounds by N contiguous SoA positions, returning 'false' if any coordinate is not finite
		// (checks all exponent bits per lane, the bounds are unspecified after non-finite coordinates)
		inline bool expand_bounds(const float* x, const float* y, const float* z, std::size_t N,
			std::array<float, 3>& minimum, std::array<float, 3>& maximum) noexcept
		{
			constexpr std::size_t N_lanes = DistQuery::N_lanes;
			std::size_t i = 0;
			if (N >= N_lanes)
			{
				float lower[3][N_lanes], upper[3][N_lanes];
#if CELLMATE_DETAIL_SIMD_WIDTH == 512
				const __m512i exponent = _mm512_set1_epi32(0x7f800000);
				// init minimums and maximums
				__m512 min_x = _mm512_set1_ps(minimum[0]);
				__m512 min_y = _mm512_set1_ps(minimum[1]);
				__m512 min_z = _mm512_set1_ps(minimum[2]);
				__m512 max_x = _mm512_set1_ps(maximum[0]);
				__m512 max_y = _mm512_set1_ps(maximum[1]);
				__m512 max_z = _mm512_set1_ps(maximum[2]);

				// check all exponent bits per lane to detect non-finite coordinates
				unsigned nonfinite = 0;
				for (; N - i >= N_lanes; i += N_lanes)
				{
					const __m512 px = _mm512_loadu_ps(x + i);
					const __m512 py = _mm512_loadu_ps(y + i);
					const __m512 pz = _mm512_loadu_ps(z + i);
					nonfinite |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(px), exponent), exponent);
					nonfinite |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(py), exponent), exponent);
					nonfinite |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(pz), exponent), exponent);
					min_x = _mm512_min_ps(min_x, px);
					min_y = _mm512_min_ps(min_y, py);
					min_z = _mm512_min_ps(min_z, pz);
					max_x = _mm512_max_ps(max_x, px);
					max_y = _mm512_max_ps(max_y, py);
					max_z = _mm512_max_ps(max_z, pz);
				}
				if (nonfinite != 0) { return false; }

				// store the per-lane results for the reduction
				_mm512_storeu_ps(lower[0], min_x);
				_mm512_storeu_ps(lower[1], min_y);
				_mm512_storeu_ps(lower[2], min_z);
				_mm512_storeu_ps(upper[0], max_x);
				_mm512_storeu_ps(upper[1], max_y);
				_mm512_storeu_ps(upper[2], max_z);
#else // CELLMATE_DETAIL_SIMD_WIDTH == 256
				const __m256i exponent = _mm256_set1_epi32(0x7f800000);
				// init minimums and maximums
				__m256 min_x = _mm256_set1_ps(minimum[0]);
				__m256 min_y = _mm256_set1_ps(minimum[1]);
				__m256 min_z = _mm256_set1_ps(minimum[2]);
				__m256 max_x = _mm256_set1_ps(maximum[0]);
				__m256 max_y = _mm256_set1_ps(maximum[1]);
				__m256 max_z = _mm256_set1_ps(maximum[2]);

				// check all exponent bits per lane to detect non-finite coordinates
				__m256i nonfinite = _mm256_setzero_si256();
				for (; N - i >= N_lanes; i += N_lanes)
				{
					const __m256 px = _mm256_loadu_ps(x + i);
					const __m256 py = _mm256_loadu_ps(y + i);
					const __m256 pz = _mm256_loadu_ps(z + i);
					nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(px), exponent), exponent));
					nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(py), exponent), exponent));
					nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(pz), exponent), exponent));
					min_x = _mm256_min_ps(min_x, px);
					min_y = _mm256_min_ps(min_y, py);
					min_z = _mm256_min_ps(min_z, pz);
					max_x = _mm256_max_ps(max_x, px);
					max_y = _mm256_max_ps(max_y, py);
					max_z = _mm256_max_ps(max_z, pz);
				}
				if (!_mm256_testz_si256(nonfinite, nonfinite)) { return false; }

				// store the per-lane results for the reduction
				_mm256_storeu_ps(lower[0], min_x);
				_mm256_storeu_ps(lower[1], min_y);
				_mm256_storeu_ps(lower[2], min_z);
				_mm256_storeu_ps(upper[0], max_x);
				_mm256_storeu_ps(upper[1], max_y);
				_mm256_storeu_ps(upper[2], max_z);
#endif
				// reduce the lanes into the min/max bounds
				for (std::size_t lane = 0; lane < N_lanes; lane++)
				{
					minimum[0] = (std::min)(minimum[0], lower[0][lane]);
					minimum[1] = (std::min)(minimum[1], lower[1][lane]);
					minimum[2] = (std::min)(minimum[2], lower[2][lane]);
					maximum[0] = (std::max)(maximum[0], upper[0][lane]);
					maximum[1] = (std::max)(maximum[1], upper[1][lane]);
					maximum[2] = (std::max)(maximum[2], upper[2][lane]);
				}
			}

			// scalar remainder
			for (; i < N; i++)
			{
				if (!std::isfinite(x[i]) ||
					!std::isfinite(y[i]) ||
					!std::isfinite(z[i]))
				{
					return false;
				}

				minimum[0] = (std::min)(minimum[0], x[i]);
				minimum[1] = (std::min)(minimum[1], y[i]);
				minimum[2] = (std::min)(minimum[2], z[i]);
				maximum[0] = (std::max)(maximum[0], x[i]);
				maximum[1] = (std::max)(maximum[1], y[i]);
				maximum[2] = (std::max)(maximum[2], z[i]);
			}
			return true;
		}

		// returns whether all N contiguous SoA positions are finite
		// (checks all exponent bits per lane)
		inline bool all_finite(const float* x, const float* y, const float* z, std::size_t N) noexcept
		{
			constexpr std::size_t N_lanes = DistQuery::N_lanes;
			std::size_t i = 0;
#if CELLMATE_DETAIL_SIMD_WIDTH == 512
			const __m512i exponent = _mm512_set1_epi32(0x7f800000);
			// check all exponent bits per lane to detect non-finite coordinates
			unsigned nonfinite = 0;
			for (; N - i >= N_lanes; i += N_lanes)
			{
				const __m512 px = _mm512_loadu_ps(x + i);
				const __m512 py = _mm512_loadu_ps(y + i);
				const __m512 pz = _mm512_loadu_ps(z + i);
				nonfinite |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(px), exponent), exponent);
				nonfinite |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(py), exponent), exponent);
				nonfinite |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(pz), exponent), exponent);
			}
			if (nonfinite != 0) { return false; }
#else // CELLMATE_DETAIL_SIMD_WIDTH == 256
			const __m256i exponent = _mm256_set1_epi32(0x7f800000);
			// check all exponent bits per lane to detect non-finite coordinates
			__m256i nonfinite = _mm256_setzero_si256();
			for (; N - i >= N_lanes; i += N_lanes)
			{
				const __m256 px = _mm256_loadu_ps(x + i);
				const __m256 py = _mm256_loadu_ps(y + i);
				const __m256 pz = _mm256_loadu_ps(z + i);
				nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(px), exponent), exponent));
				nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(py), exponent), exponent));
				nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(pz), exponent), exponent));
			}
			if (!_mm256_testz_si256(nonfinite, nonfinite)) { return false; }
#endif

			// scalar remainder
			for (; i < N; i++)
			{
				if (!std::isfinite(x[i]) ||
					!std::isfinite(y[i]) ||
					!std::isfinite(z[i]))
				{
					return false;
				}
			}
			return true;
		}
#endif

#if CELLMATE_DETAIL_SIMD_WIDTH == 512
		// positions per Morton key conversion, one fp64 lane each
		constexpr std::size_t N_key_lanes = 8;

		// converts 8 contiguous coordinates to unclamped fp64 cell coordinates relative to the grid origin
		inline __m512d cell_coords(const float* coordinates, float origin, double cell_width_inv) noexcept
		{
			CELLMATE_DETAIL_FP_NO_CONTRACT
			const __m512d offset = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(coordinates)), _mm512_set1_pd(double(origin)));
			return _mm512_roundscale_pd(_mm512_mul_pd(offset, _mm512_set1_pd(cell_width_inv)), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
		}

		// spreads 21-bit whole cell coordinates into bit positions 0, 3, 6, ..., 60 of 64-bit lanes
		inline __m512i spread_bits(__m512d cells) noexcept
		{
			__m512i bits = _mm512_cvtepi32_epi64(_mm512_cvttpd_epi32(cells));

			bits = _mm512_and_si512(_mm512_or_si512(bits, _mm512_slli_epi64(bits, 32)), _mm512_set1_epi64(0x1f00000000ffffULL));
			bits = _mm512_and_si512(_mm512_or_si512(bits, _mm512_slli_epi64(bits, 16)), _mm512_set1_epi64(0x1f0000ff0000ffULL));
			bits = _mm512_and_si512(_mm512_or_si512(bits, _mm512_slli_epi64(bits, 8)), _mm512_set1_epi64(0x100f00f00f00f00fULL));
			bits = _mm512_and_si512(_mm512_or_si512(bits, _mm512_slli_epi64(bits, 4)), _mm512_set1_epi64(0x10c30c30c30c30c3ULL));
			bits = _mm512_and_si512(_mm512_or_si512(bits, _mm512_slli_epi64(bits, 2)), _mm512_set1_epi64(0x1249249249249249ULL));

			return bits;
		}

		// interleaves three cell coordinates per lane into 63-bit Morton keys
		inline __m512i morton_keys(__m512d x, __m512d y, __m512d z) noexcept
		{
			return _mm512_or_si512(spread_bits(x),
				_mm512_or_si512(_mm512_slli_epi64(spread_bits(y), 1),
								_mm512_slli_epi64(spread_bits(z), 2)));
		}

#elif CELLMATE_DETAIL_SIMD_WIDTH == 256
		// positions per Morton key conversion, one fp64 lane each
		constexpr std::size_t N_key_lanes = 4;

		// converts 4 contiguous coordinates to unclamped fp64 cell coordinates relative to the grid origin
		inline __m256d cell_coords(const float* coordinates, float origin, double cell_width_inv) noexcept
		{
			CELLMATE_DETAIL_FP_NO_CONTRACT
			const __m256d offset = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(coordinates)), _mm256_set1_pd(double(origin)));
			return _mm256_round_pd(_mm256_mul_pd(offset, _mm256_set1_pd(cell_width_inv)), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
		}

		// spreads 21-bit whole cell coordinates into bit positions 0, 3, 6, ..., 60 of 64-bit lanes
		inline __m256i spread_bits(__m256d cells) noexcept
		{
			__m256i bits = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(cells));

			bits = _mm256_and_si256(_mm256_or_si256(bits, _mm256_slli_epi64(bits, 32)), _mm256_set1_epi64x(0x1f00000000ffffULL));
			bits = _mm256_and_si256(_mm256_or_si256(bits, _mm256_slli_epi64(bits, 16)), _mm256_set1_epi64x(0x1f0000ff0000ffULL));
			bits = _mm256_and_si256(_mm256_or_si256(bits, _mm256_slli_epi64(bits, 8)), _mm256_set1_epi64x(0x100f00f00f00f00fULL));
			bits = _mm256_and_si256(_mm256_or_si256(bits, _mm256_slli_epi64(bits, 4)), _mm256_set1_epi64x(0x10c30c30c30c30c3ULL));
			bits = _mm256_and_si256(_mm256_or_si256(bits, _mm256_slli_epi64(bits, 2)), _mm256_set1_epi64x(0x1249249249249249ULL));

			return bits;
		}

		// interleaves three cell coordinates per lane into 63-bit Morton keys
		inline __m256i morton_keys(__m256d x, __m256d y, __m256d z) noexcept
		{
			return _mm256_or_si256(spread_bits(x),
				_mm256_or_si256(_mm256_slli_epi64(spread_bits(y), 1),
								_mm256_slli_epi64(spread_bits(z), 2)));
		}
#endif

#if CELLMATE_DETAIL_SIMD_WIDTH == 512 || CELLMATE_DETAIL_SIMD_WIDTH == 256
		// converts contiguous points inside the grid into unclamped Morton keys of their cells
		// (requires 'N_key_lanes' floats per array with cell coordinates within the grid, matches the rounding of 'NeighborSearch')
		inline void point_cell_keys(const float* x, const float* y, const float* z,
			std::array<float, 3> origin, double cell_width_inv, std::uint64_t* keys) noexcept
		{
#if CELLMATE_DETAIL_SIMD_WIDTH == 512
			_mm512_storeu_si512(keys, morton_keys(cell_coords(x, origin[0], cell_width_inv),
												  cell_coords(y, origin[1], cell_width_inv),
												  cell_coords(z, origin[2], cell_width_inv)));
#else // CELLMATE_DETAIL_SIMD_WIDTH == 256
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(keys), morton_keys(cell_coords(x, origin[0], cell_width_inv),
																			  cell_coords(y, origin[1], cell_width_inv),
																			  cell_coords(z, origin[2], cell_width_inv)));
#endif
		}

		// converts contiguous queries into Morton keys of their cells clamped to the grid, with the low bit marking cells outside the grid
		// (requires 'N_key_lanes' finite floats per array, matches the rounding of 'NeighborSearch')
		inline void query_cell_keys(const float* x, const float* y, const float* z, std::array<float, 3> origin,
			double cell_width_inv, std::array<std::uint32_t, 3> last_cell, std::uint64_t* keys) noexcept
		{
#if CELLMATE_DETAIL_SIMD_WIDTH == 512
			const __m512d cell_x = cell_coords(x, origin[0], cell_width_inv);
			const __m512d cell_y = cell_coords(y, origin[1], cell_width_inv);
			const __m512d cell_z = cell_coords(z, origin[2], cell_width_inv);
			const __m512d last_x = _mm512_set1_pd(double(last_cell[0]));
			const __m512d last_y = _mm512_set1_pd(double(last_cell[1]));
			const __m512d last_z = _mm512_set1_pd(double(last_cell[2]));
			const __m512d zero = _mm512_setzero_pd();

			// check whether query's cell is outside the grid before clamping
			const __mmask8 outside_grid = static_cast<__mmask8>(
				_mm512_cmp_pd_mask(cell_x, zero, _CMP_LT_OQ) | _mm512_cmp_pd_mask(cell_x, last_x, _CMP_GT_OQ) |
				_mm512_cmp_pd_mask(cell_y, zero, _CMP_LT_OQ) | _mm512_cmp_pd_mask(cell_y, last_y, _CMP_GT_OQ) |
				_mm512_cmp_pd_mask(cell_z, zero, _CMP_LT_OQ) | _mm512_cmp_pd_mask(cell_z, last_z, _CMP_GT_OQ));

			// clamp cell coordinates to the extent of the grid
			const __m512i key = _mm512_slli_epi64(morton_keys(_mm512_max_pd(zero, _mm512_min_pd(last_x, cell_x)),
															  _mm512_max_pd(zero, _mm512_min_pd(last_y, cell_y)),
															  _mm512_max_pd(zero, _mm512_min_pd(last_z, cell_z))), 1);

			// combo of Morton key + outside-grid bit
			_mm512_storeu_si512(keys, _mm512_mask_or_epi64(key, outside_grid, key, _mm512_set1_epi64(1)));
#else // CELLMATE_DETAIL_SIMD_WIDTH == 256
			const __m256d cell_x = cell_coords(x, origin[0], cell_width_inv);
			const __m256d cell_y = cell_coords(y, origin[1], cell_width_inv);
			const __m256d cell_z = cell_coords(z, origin[2], cell_width_inv);
			const __m256d last_x = _mm256_set1_pd(double(last_cell[0]));
			const __m256d last_y = _mm256_set1_pd(double(last_cell[1]));
			const __m256d last_z = _mm256_set1_pd(double(last_cell[2]));
			const __m256d zero = _mm256_setzero_pd();

			// check whether query's cell is outside the grid before clamping
			const __m256d outside_grid = _mm256_or_pd(
				_mm256_or_pd(_mm256_or_pd(_mm256_cmp_pd(cell_x, zero, _CMP_LT_OQ), _mm256_cmp_pd(cell_x, last_x, _CMP_GT_OQ)),
							 _mm256_or_pd(_mm256_cmp_pd(cell_y, zero, _CMP_LT_OQ), _mm256_cmp_pd(cell_y, last_y, _CMP_GT_OQ))),
				_mm256_or_pd(_mm256_cmp_pd(cell_z, zero, _CMP_LT_OQ), _mm256_cmp_pd(cell_z, last_z, _CMP_GT_OQ)));

			// clamp cell coordinates to the extent of the grid
			const __m256i key = morton_keys(_mm256_max_pd(zero, _mm256_min_pd(last_x, cell_x)),
											_mm256_max_pd(zero, _mm256_min_pd(last_y, cell_y)),
											_mm256_max_pd(zero, _mm256_min_pd(last_z, cell_z)));

			// combo of Morton key + outside-grid bit
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(keys),
				_mm256_or_si256(_mm256_slli_epi64(key, 1),
				_mm256_srli_epi64(_mm256_castpd_si256(outside_grid), 63)));
#endif
		}
#endif
	}
}

CELLMATE_DETAIL_FP_END
