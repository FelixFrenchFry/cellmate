#pragma once

#include <array>
#include <cstddef>
#include <cmath>

#include "fp_control.hpp"

#if CELLMATE_DETAIL_HAS_FMA
#include <immintrin.h>
#endif

CELLMATE_DETAIL_FP_BEGIN

namespace cellmate
{
	namespace detail
	{
		// scalar backend and shared predicate for tails and indirect position access
		struct DistQueryScalar
		{
			static constexpr std::size_t N_lanes = 1;
			std::array<float, 3> query_pos_;
			float radius_squ_;

			DistQueryScalar(std::array<float, 3> query_pos, float radius_squ) noexcept
				: query_pos_(query_pos),
				radius_squ_(radius_squ)
			{}

			bool contains(float x, float y, float z) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const float dx = query_pos_[0] - x;
				const float dy = query_pos_[1] - y;
				const float dz = query_pos_[2] - z;
				float dist_squ = dx * dx;
#if CELLMATE_DETAIL_HAS_FMA
				__m128 dist_squ_simd = _mm_set_ss(dist_squ);
				dist_squ_simd = _mm_fmadd_ss(_mm_set_ss(dy), _mm_set_ss(dy), dist_squ_simd);
				dist_squ_simd = _mm_fmadd_ss(_mm_set_ss(dz), _mm_set_ss(dz), dist_squ_simd);
				dist_squ = _mm_cvtss_f32(dist_squ_simd);
#else
				dist_squ = std::fma(dy, dy, dist_squ);
				dist_squ = std::fma(dz, dz, dist_squ);
#endif
				return dist_squ <= radius_squ_;
			}

			unsigned neighbor_mask(const float* x, const float* y, const float* z) const noexcept
			{
				return static_cast<unsigned>(contains(*x, *y, *z));
			}
		};

		// scalar counterpart of the query group predicate
		struct DistQueryGroupScalar
		{
			static constexpr std::size_t N_lanes = 1;
			DistQueryScalar scalar_;

			DistQueryGroupScalar(const float* x, const float* y, const float* z, float radius_squ) noexcept
				: scalar_({ *x, *y, *z }, radius_squ)
			{}

			unsigned neighbor_mask(float px, float py, float pz) const noexcept
			{
				return static_cast<unsigned>(scalar_.contains(px, py, pz));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
