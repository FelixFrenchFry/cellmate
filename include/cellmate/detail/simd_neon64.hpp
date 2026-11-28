#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

#include "simd_scalar.hpp"

CELLMATE_DETAIL_FP_BEGIN

namespace cellmate
{
	namespace detail
	{
		inline unsigned neon64_distance_mask(float32x4_t dx, float32x4_t dy, float32x4_t dz,
			float32x4_t radius_squ) noexcept
		{
			CELLMATE_DETAIL_FP_NO_CONTRACT
			// keep the scalar sum order and rounding at the inclusive cutoff
			const float32x4_t xy_squ = vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy));
			const float32x4_t dist_squ = vaddq_f32(xy_squ, vmulq_f32(dz, dz));
			const std::uint32_t lane_bits[4]{ 1u, 2u, 4u, 8u };
			return vaddvq_u32(vandq_u32(vcleq_f32(dist_squ, radius_squ), vld1q_u32(lane_bits)));
		}

		// broadcast a query once, then test 4 contiguous candidates per call
		struct DistQueryNEON64
		{
			static constexpr std::size_t N_lanes = 4;
			DistQueryScalar scalar_;
			float32x4_t x_, y_, z_, radius_squ_;

			DistQueryNEON64(std::array<float, 3> query_pos, float radius_squ) noexcept
				: scalar_(query_pos, radius_squ),
				x_(vdupq_n_f32(query_pos[0])),
				y_(vdupq_n_f32(query_pos[1])),
				z_(vdupq_n_f32(query_pos[2])),
				radius_squ_(vdupq_n_f32(radius_squ))
			{}

			bool contains(float px, float py, float pz) const noexcept
			{
				return scalar_.contains(px, py, pz);
			}

			// callers must provide 4 valid floats per array
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				const float32x4_t dx = vsubq_f32(x_, vld1q_f32(px));
				const float32x4_t dy = vsubq_f32(y_, vld1q_f32(py));
				const float32x4_t dz = vsubq_f32(z_, vld1q_f32(pz));
				return neon64_distance_mask(dx, dy, dz, radius_squ_);
			}
		};

		// load 4 contiguous queries once, then broadcast one candidate per call
		struct DistQueryGroupNEON64
		{
			static constexpr std::size_t N_lanes = 4;
			float32x4_t x_, y_, z_, radius_squ_;

			// callers must provide 4 valid floats per array
			DistQueryGroupNEON64(const float* x, const float* y, const float* z, float radius_squ) noexcept
				: x_(vld1q_f32(x)),
				y_(vld1q_f32(y)),
				z_(vld1q_f32(z)),
				radius_squ_(vdupq_n_f32(radius_squ))
			{}

			unsigned neighbor_mask(float px, float py, float pz) const noexcept
			{
				const float32x4_t dx = vsubq_f32(x_, vdupq_n_f32(px));
				const float32x4_t dy = vsubq_f32(y_, vdupq_n_f32(py));
				const float32x4_t dz = vsubq_f32(z_, vdupq_n_f32(pz));
				return neon64_distance_mask(dx, dy, dz, radius_squ_);
			}
		};

		// broadcast the query group's bounding box once, then test 4 contiguous candidates per call
		struct DistBoxNEON64
		{
			static constexpr std::size_t N_lanes = 4;
			float32x4_t first_x_, first_y_, first_z_, last_x_, last_y_, last_z_, radius_squ_;

			DistBoxNEON64(std::array<float, 3> first, std::array<float, 3> last, float radius_squ) noexcept
				: first_x_(vdupq_n_f32(first[0])),
				first_y_(vdupq_n_f32(first[1])),
				first_z_(vdupq_n_f32(first[2])),
				last_x_(vdupq_n_f32(last[0])),
				last_y_(vdupq_n_f32(last[1])),
				last_z_(vdupq_n_f32(last[2])),
				radius_squ_(vdupq_n_f32(radius_squ))
			{}

			// callers must provide 4 valid floats per array
			unsigned neighbor_mask(const float* px, const float* py, const float* pz) const noexcept
			{
				const float32x4_t x = vld1q_f32(px);
				const float32x4_t y = vld1q_f32(py);
				const float32x4_t z = vld1q_f32(pz);
				const float32x4_t zero = vdupq_n_f32(0.0f);
				const float32x4_t dx = vmaxq_f32(vmaxq_f32(vsubq_f32(first_x_, x), vsubq_f32(x, last_x_)), zero);
				const float32x4_t dy = vmaxq_f32(vmaxq_f32(vsubq_f32(first_y_, y), vsubq_f32(y, last_y_)), zero);
				const float32x4_t dz = vmaxq_f32(vmaxq_f32(vsubq_f32(first_z_, z), vsubq_f32(z, last_z_)), zero);
				return neon64_distance_mask(dx, dy, dz, radius_squ_);
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
