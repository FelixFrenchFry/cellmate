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
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m256 dx = _mm256_sub_ps(x_, _mm256_loadu_ps(px));
				const __m256 dy = _mm256_sub_ps(y_, _mm256_loadu_ps(py));
				const __m256 dz = _mm256_sub_ps(z_, _mm256_loadu_ps(pz));
				// preserve the scalar predicate's fused operation order
				__m256 dist_squ = _mm256_mul_ps(dx, dx);
				dist_squ = _mm256_fmadd_ps(dy, dy, dist_squ);
				dist_squ = _mm256_fmadd_ps(dz, dz, dist_squ);
				return static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(dist_squ, radius_squ_, _CMP_LE_OQ)));
			}
		};

		// load contiguous 8 query positions once, then broadcast one candidate per call
		struct DistQueryGroupAVX256
		{
			static constexpr std::size_t N_lanes = 8;
			__m256 x_, y_, z_, radius_squ_;

			// callers must provide 8 valid floats per array
			DistQueryGroupAVX256(const float* x, const float* y, const float* z, float radius_squ) noexcept
				: x_(_mm256_loadu_ps(x)),
				y_(_mm256_loadu_ps(y)),
				z_(_mm256_loadu_ps(z)),
				radius_squ_(_mm256_set1_ps(radius_squ))
			{}

			unsigned neighbor_mask(float px, float py, float pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m256 dx = _mm256_sub_ps(x_, _mm256_set1_ps(px));
				const __m256 dy = _mm256_sub_ps(y_, _mm256_set1_ps(py));
				const __m256 dz = _mm256_sub_ps(z_, _mm256_set1_ps(pz));
				// preserve the scalar predicate's fused operation order
				__m256 dist_squ = _mm256_mul_ps(dx, dx);
				dist_squ = _mm256_fmadd_ps(dy, dy, dist_squ);
				dist_squ = _mm256_fmadd_ps(dz, dz, dist_squ);
				return static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(dist_squ, radius_squ_, _CMP_LE_OQ)));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
