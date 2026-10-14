#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../include/cellmate/detail/fp_control.hpp"

CELLMATE_DETAIL_FP_BEGIN

namespace cellmate_helpers
{
	// expects finite positions and a finite positive radius
	inline void brute_force(const float* x, const float* y, const float* z,
		std::size_t N_points, float radius, std::array<float, 3> query_pos,
		std::vector<std::uint32_t>& neighbors)
	{
		CELLMATE_DETAIL_FP_NO_CONTRACT
		neighbors.clear();

		if (N_points > (std::numeric_limits<std::uint32_t>::max)())
		{
			throw std::length_error("'cellmate_helpers::brute_force()' 'N_points' exceeds 32-bit point count limits");
		}
		const std::uint32_t N_points_32 = static_cast<std::uint32_t>(N_points);
		const float radius_squ = radius * radius;

		// check every point, including self and points on the cutoff
		for (std::uint32_t i = 0; i < N_points_32; i++)
		{
			const float dx = query_pos[0] - x[i];
			const float dy = query_pos[1] - y[i];
			const float dz = query_pos[2] - z[i];
			
			if (dx * dx + dy * dy + dz * dz <= radius_squ)
			{
				neighbors.push_back(i);
			}
		}
	}
}

CELLMATE_DETAIL_FP_END
