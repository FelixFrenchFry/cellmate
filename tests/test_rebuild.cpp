#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// verifies query results and cache invalidation after rebuilds
// (moves points and changes point count between rebuilds)
TEST(NeighborSearchRebuild, RebuildMatchesBruteForce)
{
	const float radius = 1.0f;
	cellmate::NeighborSearch search(radius);
	std::vector<std::size_t> neighbors, expected;
	float x[]{ 0.0f, 4.0f, 12.0f };
	const float y[]{ 0.0f, 0.0f, 0.0f };
	const float z[]{ 0.0f, 0.0f, 0.0f };
	const float queries[]{ 4.0f, 8.0f, 12.0f, 12.0f };

	for (std::size_t pass = 0; pass < 3; pass++)
	{
		x[1] = 4.0f * float(pass + 1);
		const std::size_t count = pass == 2 ? 2 : 3;
		search.rebuild(x, y, z, count);

		for (float qx : queries)
		{
			search.query({ qx, 0.0f, 0.0f }, neighbors);
			cellmate_helpers::brute_force(x, y, z, count, radius, { qx, 0.0f, 0.0f }, expected);

			std::sort(neighbors.begin(), neighbors.end());
			ASSERT_EQ(neighbors, expected) << "rebuild pass = " << pass << ", query x = " << qx;
		}
	}
}

// verifies that failed rebuilds clear query results and allow reuse
// (also checks empty rebuilds in both storage and execution modes)
TEST(NeighborSearchRebuild, FailedAndEmptyRebuildsClearTheSearch)
{
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			auto points = cellmate_helpers::make_uniform_cloud(10003, 91, -5.0f, 5.0f);
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), execution);
			const float original_x = points.x[0];
			points.x[0] = std::numeric_limits<float>::quiet_NaN();
			EXPECT_THROW(search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), execution),
				std::invalid_argument);
			points.x[0] = original_x;
			std::vector<std::size_t> neighbors, offsets;
			search.query({ original_x, points.y[0], points.z[0] }, neighbors);
			ASSERT_TRUE(neighbors.empty());
			search.query_batch(points.x.data(), points.y.data(), points.z.data(), 1, neighbors, offsets, execution);
			ASSERT_TRUE(neighbors.empty());
			ASSERT_EQ(offsets, (std::vector<std::size_t>{ 0, 0 }));

			search.rebuild(points.x.data(), points.y.data(), points.z.data(), 1, execution);
			search.query({ original_x, points.y[0], points.z[0] }, neighbors);
			ASSERT_EQ(neighbors, (std::vector<std::size_t>{ 0 }));
			search.rebuild(nullptr, nullptr, nullptr, 0, execution);
			search.query({ original_x, points.y[0], points.z[0] }, neighbors);
			ASSERT_TRUE(neighbors.empty());
		}
	}
}
