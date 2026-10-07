#include <cellmate/neighbor_search.hpp>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
	const std::size_t N_points = 10000;
	const float radius = 1.0f;
	const cellmate_helpers::PointCloud points = cellmate_helpers::make_uniform_cloud(N_points, 67, -5.0f, 5.0f);

	cellmate::NeighborSearch search(radius);
	search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points);

	// verifies that the neighbor search result matches the brute force approach
	std::vector<std::size_t> neighbors, expected;
	for (std::size_t i = 0; i < N_points; i++)
	{
		const std::array<float, 3> query_pos{ points.x[i], points.y[i], points.z[i] };
		search.query(query_pos, neighbors);

		cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(),
			N_points, radius, query_pos, expected);

		// (output order of the neighbor search result does not matter)
		std::sort(neighbors.begin(), neighbors.end());
		if (neighbors != expected)
		{
			std::cerr << "Neighbor mismatch at point " << i << '\n';
			return 1;
		}
	}

	float x[]{ 0.0f, 4.0f, 12.0f };
	const float y[]{ 0.0f, 0.0f, 0.0f };
	const float z[]{ 0.0f, 0.0f, 0.0f };
	const float queries[]{ 4.0f, 8.0f, 12.0f, 12.0f };

	// verifies cache reuse and invalidation after rebuilds
	for (std::size_t pass = 0; pass < 3; pass++)
	{
		x[1] = 4.0f * float(pass + 1);
		const std::size_t count = pass == 2 ? 2 : 3;
		search.rebuild(x, y, z, count);

		for (float qx : queries)
		{
			search.query({ qx, 0.0f, 0.0f }, neighbors);
			cellmate_helpers::brute_force(x, y, z, count, radius, { qx, 0.0f, 0.0f }, expected);

			// (output order of the neighbor search result does not matter)
			std::sort(neighbors.begin(), neighbors.end());
			if (neighbors != expected)
			{
				std::cerr << "Neighbor mismatch after rebuild " << pass << " at x = " << qx << '\n';
				return 1;
			}
		}
	}

	std::cout << "All " << N_points << " CLL queries and cache checks match brute force\n";
	return 0;
}
