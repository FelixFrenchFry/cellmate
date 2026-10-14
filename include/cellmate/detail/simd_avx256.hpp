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
				// (omitting FMA to match rounding precision of the scalar version, despite potential accuracy gains)
				const __m256 dist_squ = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx),
																		_mm256_mul_ps(dy, dy)),
																		_mm256_mul_ps(dz, dz));
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
				// (omitting FMA to match rounding precision of the scalar version, despite potential accuracy gains)
				const __m256 dist_squ = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx),
																		_mm256_mul_ps(dy, dy)),
																		_mm256_mul_ps(dz, dz));
				return static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(dist_squ, radius_squ_, _CMP_LE_OQ)));
			}
		};

		// broadcast a query bounding box once, then test 8 contiguous candidates per call
		// (never rejects a candidate accepted by 'DistQueryAVX256' for a query inside the box, by monotonic rounding)
		struct DistBoxAVX256
		{
			static constexpr std::size_t N_lanes = 8;
			__m256 first_x_, first_y_, first_z_, last_x_, last_y_, last_z_, radius_squ_;

			DistBoxAVX256(std::array<float, 3> first, std::array<float, 3> last, float radius_squ) noexcept
				: first_x_(_mm256_set1_ps(first[0])),
				first_y_(_mm256_set1_ps(first[1])),
				first_z_(_mm256_set1_ps(first[2])),
				last_x_(_mm256_set1_ps(last[0])),
				last_y_(_mm256_set1_ps(last[1])),
				last_z_(_mm256_set1_ps(last[2])),
				radius_squ_(_mm256_set1_ps(radius_squ))
			{}

			// callers must provide 8 valid floats per array
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const __m256 x = _mm256_loadu_ps(px);
				const __m256 y = _mm256_loadu_ps(py);
				const __m256 z = _mm256_loadu_ps(pz);
				// distance to the bounding box along each axis, zero inside
				const __m256 zero = _mm256_setzero_ps();
				const __m256 dx = _mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(first_x_, x), _mm256_sub_ps(x, last_x_)), zero);
				const __m256 dy = _mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(first_y_, y), _mm256_sub_ps(y, last_y_)), zero);
				const __m256 dz = _mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(first_z_, z), _mm256_sub_ps(z, last_z_)), zero);
				const __m256 dist_squ = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx),
																	_mm256_mul_ps(dy, dy)),
																	_mm256_mul_ps(dz, dz));
				return static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(dist_squ, radius_squ_, _CMP_LE_OQ)));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
