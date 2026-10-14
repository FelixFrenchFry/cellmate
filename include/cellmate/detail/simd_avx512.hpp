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
				// (omitting FMA to match rounding precision of the scalar version, despite potential accuracy gains)
				const __m512 dist_squ = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(dx, dx),
																		_mm512_mul_ps(dy, dy)),
																		_mm512_mul_ps(dz, dz));
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
				// (omitting FMA to match rounding precision of the scalar version, despite potential accuracy gains)
				const __m512 dist_squ = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(dx, dx),
																		_mm512_mul_ps(dy, dy)),
																		_mm512_mul_ps(dz, dz));
				return static_cast<unsigned>(_mm512_cmp_ps_mask(dist_squ, radius_squ_, _CMP_LE_OQ));
			}
		};

		// broadcast a query bounding box once, then test 16 contiguous candidates per call
		// (never rejects a candidate accepted by 'DistQueryAVX512' for a query inside the box, by monotonic rounding)
		struct DistBoxAVX512
		{
			static constexpr std::size_t N_lanes = 16;
			__m512 first_x_, first_y_, first_z_, last_x_, last_y_, last_z_, radius_squ_;

			DistBoxAVX512(std::array<float, 3> first, std::array<float, 3> last, float radius_squ) noexcept
				: first_x_(_mm512_set1_ps(first[0])),
				first_y_(_mm512_set1_ps(first[1])),
				first_z_(_mm512_set1_ps(first[2])),
				last_x_(_mm512_set1_ps(last[0])),
				last_y_(_mm512_set1_ps(last[1])),
				last_z_(_mm512_set1_ps(last[2])),
				radius_squ_(_mm512_set1_ps(radius_squ))
			{}

			// callers must provide 16 valid floats per array
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m512 x = _mm512_loadu_ps(px);
				const __m512 y = _mm512_loadu_ps(py);
				const __m512 z = _mm512_loadu_ps(pz);
				// distance to the bounding box along each axis, zero inside
				const __m512 zero = _mm512_setzero_ps();
				const __m512 dx = _mm512_max_ps(_mm512_max_ps(_mm512_sub_ps(first_x_, x), _mm512_sub_ps(x, last_x_)), zero);
				const __m512 dy = _mm512_max_ps(_mm512_max_ps(_mm512_sub_ps(first_y_, y), _mm512_sub_ps(y, last_y_)), zero);
				const __m512 dz = _mm512_max_ps(_mm512_max_ps(_mm512_sub_ps(first_z_, z), _mm512_sub_ps(z, last_z_)), zero);
				const __m512 dist_squ = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(dx, dx),
																	_mm512_mul_ps(dy, dy)),
																	_mm512_mul_ps(dz, dz));
				return static_cast<unsigned>(_mm512_cmp_ps_mask(dist_squ, radius_squ_, _CMP_LE_OQ));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
