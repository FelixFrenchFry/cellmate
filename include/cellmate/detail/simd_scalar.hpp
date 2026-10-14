#pragma once

#include <array>
#include <cstddef>

#include "fp_control.hpp"

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
				return dx * dx + dy * dy + dz * dz <= radius_squ_;
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

		// tests whether a candidate is within the radius of any query inside an axis-aligned box, filtering them out
		// (never rejects a candidate accepted by 'DistQueryScalar' for a query inside the box, by monotonic rounding)
		struct DistBoxScalar
		{
			static constexpr std::size_t N_lanes = 1;
			std::array<float, 3> first_;
			std::array<float, 3> last_;
			float radius_squ_;

			DistBoxScalar(std::array<float, 3> first, std::array<float, 3> last, float radius_squ) noexcept
				: first_(first),
				last_(last),
				radius_squ_(radius_squ)
			{}

			bool contains(float x, float y, float z) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				// distance to the bounding box along each axis, zero inside
				const float dx = first_[0] - x > 0.0f ? first_[0] - x : (x - last_[0] > 0.0f ? x - last_[0] : 0.0f);
				const float dy = first_[1] - y > 0.0f ? first_[1] - y : (y - last_[1] > 0.0f ? y - last_[1] : 0.0f);
				const float dz = first_[2] - z > 0.0f ? first_[2] - z : (z - last_[2] > 0.0f ? z - last_[2] : 0.0f);
				return dx * dx + dy * dy + dz * dz <= radius_squ_;
			}

			unsigned neighbor_mask(const float* x, const float* y, const float* z) const noexcept
			{
				return static_cast<unsigned>(contains(*x, *y, *z));
			}
		};
	}
}

CELLMATE_DETAIL_FP_END
