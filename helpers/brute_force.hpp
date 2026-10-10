#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "../include/cellmate/detail/fp_control.hpp"

CELLMATE_DETAIL_FP_BEGIN

namespace cellmate_helpers
{
	// expects finite positions and a finite positive radius
	inline void brute_force(const float* x, const float* y, const float* z,
		std::size_t N_points, float radius, std::array<float, 3> query_pos,
		std::vector<std::size_t>& neighbors)
	{
		CELLMATE_DETAIL_FP_NO_CONTRACT
		neighbors.clear();
		const float radius_sq = radius * radius;

		// check every point, including self and points on the cutoff
		for (std::size_t i = 0; i < N_points; i++)
		{
			const float dx = query_pos[0] - x[i];
			const float dy = query_pos[1] - y[i];
			const float dz = query_pos[2] - z[i];
			
			if (dx * dx + dy * dy + dz * dz <= radius_sq)
			{
				neighbors.push_back(i);
			}
		}
	}
}

CELLMATE_DETAIL_FP_END
