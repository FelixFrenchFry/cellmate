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
			double radius_squ_;

			DistQueryScalar(std::array<float, 3> query_pos, double radius_squ) noexcept
				: query_pos_(query_pos),
				radius_squ_(radius_squ)
			{}

			bool contains(float x, float y, float z) const noexcept
			{
				CELLMATE_DETAIL_FP_NO_CONTRACT
				const double dx = double(query_pos_[0]) - x;
				const double dy = double(query_pos_[1]) - y;
				const double dz = double(query_pos_[2]) - z;
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
