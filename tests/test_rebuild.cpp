#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
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

// verifies sparse cell lookups across large, small, and empty rebuilds
// (covers empty regions, both storage modes, and sequential and parallel batches)
TEST(NeighborSearchRebuild, SparseCellsMatchAcrossRebuilds)
{
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			cellmate::NeighborSearch search(1.0f, storage);
			cellmate_helpers::PointCloud points;
			std::vector<std::size_t> neighbors, offsets, expected;
			for (std::size_t N_points : { 2047u, 2048u, 2049u, 8193u, 1000u, 1u, 0u, 2048u })
			{
				SCOPED_TRACE(::testing::Message() << "point count = " << N_points
					<< ", storage = " << static_cast<int>(storage) << ", execution = " << static_cast<int>(execution));
				// alternate spacing so smaller rebuilds move cell keys to different indices
				const float spacing = N_points % 2 != 0 ? 3.0f : 6.0f;
				points.x.resize(N_points);
				points.y.assign(N_points, 0.0f);
				points.z.assign(N_points, 0.0f);
				for (std::size_t i = 0; i < N_points; i++) { points.x[i] = spacing * float(i); }
				const float extent = spacing * float(N_points);
				if (N_points != 0) { points.y.back() = points.z.back() = extent; }
				search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points, execution);

				const std::array<float, 3> positions[]{
					{ 0.0f, 0.0f, 0.0f }, { extent / 2, 0.0f, 0.0f }, { 1.5f, 0.0f, 0.0f },
					{ extent / 2, extent / 2, 0.0f }, { extent - spacing, extent, extent }
				};
				for (const auto& position : positions)
				{
					search.query(position, neighbors);
					cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(),
						N_points, 1.0f, position, expected);
					std::sort(neighbors.begin(), neighbors.end());
					ASSERT_EQ(neighbors, expected);
				}

				// separated points have exactly one neighbor in each query row
				search.query_batch(points.x.data(), points.y.data(), points.z.data(), N_points,
					neighbors, offsets, execution);
				ASSERT_EQ(neighbors.size(), N_points);
				ASSERT_EQ(offsets.size(), N_points + 1);
				for (std::size_t i = 0; i < N_points; i++)
				{
					ASSERT_EQ(offsets[i], i);
					ASSERT_EQ(neighbors[i], i);
				}
				ASSERT_EQ(offsets.back(), N_points);
			}
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
