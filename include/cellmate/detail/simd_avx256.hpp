#pragma once

#include <array>
#include <cstddef>
#include <immintrin.h>

#include "simd_scalar.hpp"

CELLMATE_DETAIL_FP_BEGIN

namespace cellmate
{
	namespace detail
	{
		// broadcast query values once, then test 4 contiguous candidates per call
		struct DistQueryAVX256
		{
			static constexpr std::size_t N_lanes = 4;
			DistQueryScalar scalar_;
			__m256d x_, y_, z_, radius_squ_;

			DistQueryAVX256(std::array<float, 3> query_pos, double radius_squ) noexcept
				: scalar_(query_pos, radius_squ),
				x_(_mm256_set1_pd(query_pos[0])),
				y_(_mm256_set1_pd(query_pos[1])),
				z_(_mm256_set1_pd(query_pos[2])),
				radius_squ_(_mm256_set1_pd(radius_squ))
			{}

			bool contains(float px, float py, float pz) const noexcept
			{
				return scalar_.contains(px, py, pz);
			}

			// callers provide 4 valid floats per array
			// (cell starts not required to be aligned)
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m256d dx = _mm256_sub_pd(x_, _mm256_cvtps_pd(_mm_loadu_ps(px)));
				const __m256d dy = _mm256_sub_pd(y_, _mm256_cvtps_pd(_mm_loadu_ps(py)));
				const __m256d dz = _mm256_sub_pd(z_, _mm256_cvtps_pd(_mm_loadu_ps(pz)));
				// (omitting FMA to match rounding precision of the scalar version, despite potential accuracy gains)
				const __m256d distance_squ = _mm256_add_pd(
					_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)), _mm256_mul_pd(dz, dz));
				return static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(distance_squ, radius_squ_, _CMP_LE_OQ)));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
