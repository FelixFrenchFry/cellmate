#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

// verifies query results against brute force without changing the input arrays
// (covers both storage and execution modes, duplicates, and radius boundaries)
TEST(NeighborSearchQueryBatch, BothModesPreserveInputsAndMatchBruteForce)
{
	cellmate_helpers::PointCloud points = cellmate_helpers::make_uniform_cloud(10003, 91, -5.0f, 5.0f);
	points.x[0] = points.x[1] = 0.0f;
	points.x[2] = 1.0f;
	points.x[3] = std::nextafter(1.0f, 2.0f);
	for (std::size_t i = 0; i < 4; i++) { points.y[i] = points.z[i] = 0.0f; }
	const auto original = points;
	const auto queries = cellmate_helpers::make_uniform_cloud(513, 67, -6.0f, 6.0f);

	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			SCOPED_TRACE(::testing::Message() << "storage = " << static_cast<int>(storage)
				<< ", execution = " << static_cast<int>(execution));
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), execution);
			ASSERT_EQ(points.x, original.x);
			ASSERT_EQ(points.y, original.y);
			ASSERT_EQ(points.z, original.z);

			std::vector<std::size_t> neighbors, offsets, single, expected;
			search.query({ 0.0f, 0.0f, 0.0f }, single);
			cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), points.x.size(),
				1.0f, { 0.0f, 0.0f, 0.0f }, expected);
			std::sort(single.begin(), single.end());
			ASSERT_EQ(single, expected);
			search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), queries.x.size(),
				neighbors, offsets, execution);
			ASSERT_EQ(offsets.size(), queries.x.size() + 1);
			ASSERT_EQ(offsets.front(), 0);
			ASSERT_EQ(offsets.back(), neighbors.size());
			for (std::size_t i = 0; i < queries.x.size(); i++)
			{
				ASSERT_LE(offsets[i], offsets[i + 1]);
				ASSERT_LE(offsets[i + 1], neighbors.size());
				const std::array<float, 3> query_pos{ queries.x[i], queries.y[i], queries.z[i] };
				search.query(query_pos, single);
				ASSERT_EQ(single.size(), offsets[i + 1] - offsets[i]);
				ASSERT_TRUE(std::equal(single.begin(), single.end(), neighbors.begin() + offsets[i]));
				cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), points.x.size(),
					1.0f, query_pos, expected);
				std::sort(single.begin(), single.end());
				ASSERT_EQ(single, expected) << "query index = " << i;
			}
		}
	}
}
