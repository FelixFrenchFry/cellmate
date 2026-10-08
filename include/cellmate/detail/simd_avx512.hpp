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
		// broadcast query values once, then test 8 contiguous candidates per call
		struct DistQueryAVX512
		{
			static constexpr std::size_t N_lanes = 8;
			DistQueryScalar scalar_;
			__m512d x_, y_, z_, radius_squ_;

			DistQueryAVX512(std::array<float, 3> query_pos, double radius_squ) noexcept
				: scalar_(query_pos, radius_squ),
				x_(_mm512_set1_pd(query_pos[0])),
				y_(_mm512_set1_pd(query_pos[1])),
				z_(_mm512_set1_pd(query_pos[2])),
				radius_squ_(_mm512_set1_pd(radius_squ))
			{}

			bool contains(float px, float py, float pz) const noexcept
			{
				return scalar_.contains(px, py, pz);
			}

			// callers provide 8 valid floats per array
			// (cell starts not required to be aligned)
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m512d dx = _mm512_sub_pd(x_, _mm512_cvtps_pd(_mm256_loadu_ps(px)));
				const __m512d dy = _mm512_sub_pd(y_, _mm512_cvtps_pd(_mm256_loadu_ps(py)));
				const __m512d dz = _mm512_sub_pd(z_, _mm512_cvtps_pd(_mm256_loadu_ps(pz)));
				// (omitting FMA to match rounding precision of the scalar version, despite potential accuracy gains)
				const __m512d distance_squ = _mm512_add_pd(
					_mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)), _mm512_mul_pd(dz, dz));
				return static_cast<unsigned>(_mm512_cmp_pd_mask(distance_squ, radius_squ_, _CMP_LE_OQ));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
