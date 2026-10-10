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
		struct DistQueryAVX256
		{
			static constexpr std::size_t N_lanes = 8;
			DistQueryScalar scalar_;
			__m256 x_, y_, z_, radius_squ_;

			DistQueryAVX256(std::array<float, 3> query_pos, float radius_squ) noexcept
				: scalar_(query_pos, radius_squ),
				x_(_mm256_set1_ps(query_pos[0])),
				y_(_mm256_set1_ps(query_pos[1])),
				z_(_mm256_set1_ps(query_pos[2])),
				radius_squ_(_mm256_set1_ps(radius_squ))
			{}

			bool contains(float px, float py, float pz) const noexcept
			{
				return scalar_.contains(px, py, pz);
			}

			// callers must provide 8 valid floats per array
			// (cell starts not required to be aligned)
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m256 dx = _mm256_sub_ps(x_, _mm256_loadu_ps(px));
				const __m256 dy = _mm256_sub_ps(y_, _mm256_loadu_ps(py));
				const __m256 dz = _mm256_sub_ps(z_, _mm256_loadu_ps(pz));
				// (omitting FMA to match rounding precision of the scalar version, despite potential accuracy gains)
				const __m256 distance_squ = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx),
																		_mm256_mul_ps(dy, dy)),
																		_mm256_mul_ps(dz, dz));
				return static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(distance_squ, radius_squ_, _CMP_LE_OQ)));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
