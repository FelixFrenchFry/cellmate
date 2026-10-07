#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace cellmate_helpers
{
	// expects finite positions and a finite positive radius
	inline void brute_force(const float* x, const float* y, const float* z,
		std::size_t N_points, float radius, std::array<float, 3> query_pos,
		std::vector<std::size_t>& neighbors)
	{
		neighbors.clear();
		const double radius_sq = double(radius) * radius;

		// check every point, including self and points on the cutoff
		for (std::size_t i = 0; i < N_points; i++)
		{
			const double dx = double(query_pos[0]) - x[i];
			const double dy = double(query_pos[1]) - y[i];
			const double dz = double(query_pos[2]) - z[i];
			
			if (dx * dx + dy * dy + dz * dz <= radius_sq)
			{
				neighbors.push_back(i);
			}
		}
	}
}
