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
		// broadcast query values once, then test 16 contiguous candidates per call
		struct DistQueryAVX512
		{
			static constexpr std::size_t N_lanes = 16;
			DistQueryScalar scalar_;
			__m512 x_, y_, z_, radius_squ_;

			DistQueryAVX512(std::array<float, 3> query_pos, float radius_squ) noexcept
				: scalar_(query_pos, radius_squ),
				x_(_mm512_set1_ps(query_pos[0])),
				y_(_mm512_set1_ps(query_pos[1])),
				z_(_mm512_set1_ps(query_pos[2])),
				radius_squ_(_mm512_set1_ps(radius_squ))
			{}

			bool contains(float px, float py, float pz) const noexcept
			{
				return scalar_.contains(px, py, pz);
			}

			// callers must provide 16 valid floats per array
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m512 dx = _mm512_sub_ps(x_, _mm512_loadu_ps(px));
				const __m512 dy = _mm512_sub_ps(y_, _mm512_loadu_ps(py));
				const __m512 dz = _mm512_sub_ps(z_, _mm512_loadu_ps(pz));
				// preserve the scalar predicate's fused operation order
				__m512 dist_squ = _mm512_mul_ps(dx, dx);
				dist_squ = _mm512_fmadd_ps(dy, dy, dist_squ);
				dist_squ = _mm512_fmadd_ps(dz, dz, dist_squ);
				return static_cast<unsigned>(_mm512_cmp_ps_mask(dist_squ, radius_squ_, _CMP_LE_OQ));
			}
		};

		// load contiguous 16 query positions once, then broadcast one candidate per call
		struct DistQueryGroupAVX512
		{
			static constexpr std::size_t N_lanes = 16;
			__m512 x_, y_, z_, radius_squ_;

			// callers must provide 16 valid floats per array
			DistQueryGroupAVX512(const float* x, const float* y, const float* z, float radius_squ) noexcept
				: x_(_mm512_loadu_ps(x)),
				y_(_mm512_loadu_ps(y)),
				z_(_mm512_loadu_ps(z)),
				radius_squ_(_mm512_set1_ps(radius_squ))
			{}

			unsigned neighbor_mask(float px, float py, float pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m512 dx = _mm512_sub_ps(x_, _mm512_set1_ps(px));
				const __m512 dy = _mm512_sub_ps(y_, _mm512_set1_ps(py));
				const __m512 dz = _mm512_sub_ps(z_, _mm512_set1_ps(pz));
				// preserve the scalar predicate's fused operation order
				__m512 dist_squ = _mm512_mul_ps(dx, dx);
				dist_squ = _mm512_fmadd_ps(dy, dy, dist_squ);
				dist_squ = _mm512_fmadd_ps(dz, dz, dist_squ);
				return static_cast<unsigned>(_mm512_cmp_ps_mask(dist_squ, radius_squ_, _CMP_LE_OQ));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
