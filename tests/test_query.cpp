#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <thread>
#include <vector>

// verifies that queries at every point match brute force
// (output order of the neighbor search result does not matter)
TEST(NeighborSearchQuery, QueryMatchesBruteForce)
{
	const std::size_t N_points = 10000;
	const float radius = 1.0f;
	const cellmate_helpers::PointCloud points = cellmate_helpers::make_uniform_cloud(N_points, 67, -5.0f, 5.0f);

	cellmate::NeighborSearch search(radius);
	search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points);

	std::vector<std::size_t> neighbors, expected;
	for (std::size_t i = 0; i < N_points; i++)
	{
		const std::array<float, 3> query_pos{ points.x[i], points.y[i], points.z[i] };
		search.query(query_pos, neighbors);

		cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(),
			N_points, radius, query_pos, expected);

		std::sort(neighbors.begin(), neighbors.end());
		ASSERT_EQ(neighbors, expected) << "seed = 67, query index = " << i;
	}
}

// verifies that concurrent queries on a shared search match brute force
// (each thread uses its own output vector, in both storage modes)
TEST(NeighborSearchQuery, ConcurrentQueriesMatchBruteForce)
{
	const auto points = cellmate_helpers::make_uniform_cloud(10003, 91, -5.0f, 5.0f);
	std::vector<std::vector<std::size_t>> expected(32);
	for (std::size_t i = 0; i < expected.size(); i++)
	{
		cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), points.x.size(),
			1.0f, { points.x[i], points.y[i], points.z[i] }, expected[i]);
	}
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size());
		std::array<bool, 2> valid{ true, true };
		std::array<std::thread, 2> workers;
		for (std::size_t worker = 0; worker < workers.size(); worker++)
		{
			workers[worker] = std::thread([&, worker]()
			{
				std::vector<std::size_t> neighbors;
				for (std::size_t i = 0; i < expected.size(); i++)
				{
					search.query({ points.x[i], points.y[i], points.z[i] }, neighbors);
					std::sort(neighbors.begin(), neighbors.end());
					if (neighbors != expected[i]) { valid[worker] = false; }
				}
			});
		}
		for (auto& worker : workers) { worker.join(); }
		for (bool result : valid) { ASSERT_TRUE(result); }
	}
}
