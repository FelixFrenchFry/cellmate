#include <cellmate/neighbor_search.hpp>

#include "brute_force.hpp"
#include "point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
	const std::size_t N_points = 10000;
	const float radius = 1.0f;
	const cellmate_test::PointCloud points = cellmate_test::make_uniform_cloud(N_points, 67, -5.0f, 5.0f);

	cellmate::NeighborSearch search(radius);
	search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points);

	std::vector<std::size_t> neighbors, expected;
	for (std::size_t i = 0; i < N_points; i++)
	{
		const std::array<float, 3> query_pos{ points.x[i], points.y[i], points.z[i] };
		search.find_neighbors(query_pos, neighbors);

		cellmate_test::brute_force(points.x.data(), points.y.data(), points.z.data(),
			N_points, radius, query_pos, expected);

		// compare neighbor indices to expected brute force result
		// (output order of the neighbor search result does not matter)
		std::sort(neighbors.begin(), neighbors.end());
		if (neighbors != expected)
		{
			std::cerr << "Neighbor mismatch at point " << i << '\n';
			return 1;
		}
	}

	std::cout << "All " << N_points << " CLL queries match brute force\n";
	return 0;
}
