#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

#ifdef _MSC_VER
// verifies that linking cellmate and including its headers preserve strict rounding
// (the test target uses '/fp:strict' on MSVC)
TEST(NeighborSearchQuery, PreservesConsumerRounding)
{
	const int original_rounding = std::fegetround();
	ASSERT_NE(original_rounding, -1);
	ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
	const double value = 1.0 + 1.1102230246251565404236316680908203125e-16;
	const int restored = std::fesetround(original_rounding);
	ASSERT_EQ(restored, 0);
	ASSERT_GT(value, 1.0);
}
#endif

// verifies that squared radii stay within the supported float range
TEST(NeighborSearchQuery, RejectsUnsupportedSquaredRadii)
{
	const float smallest = std::ldexp(1.0f, -63);
	const float largest = std::nextafter(std::ldexp(1.0f, 64), 0.0f);
	EXPECT_NO_THROW((void)cellmate::NeighborSearch(smallest));
	EXPECT_NO_THROW((void)cellmate::NeighborSearch(largest));
	for (float radius : { 0.0f, -1.0f, std::nextafter(smallest, 0.0f),
		std::ldexp(1.0f, 64), (std::numeric_limits<float>::max)(),
		(std::numeric_limits<float>::min)(), std::numeric_limits<float>::denorm_min(),
		std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() })
	{
		SCOPED_TRACE(::testing::Message() << "'radius' = " << radius);
		EXPECT_THROW((void)cellmate::NeighborSearch(radius), std::invalid_argument);
	}
}

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

// verifies float cell bounds near the grid limit and at supported radius extremes
TEST(NeighborSearchQuery, FloatCellBoundsMatchBruteForceAcrossWideExtents)
{
	for (float radius : { std::ldexp(1.0f, -63), 0.75f, 1.0f,
		std::nextafter(std::ldexp(1.0f, 64), 0.0f) })
	{
		cellmate_helpers::PointCloud points{ { 0.0f }, { 0.0f }, { 0.0f } };
		const float center = radius * float((1u << 21) - 16);
		for (int i = -12; i <= 12; i++)
		{
			const float coordinate = center + float(i) * 0.25f * radius;
			points.x.push_back(coordinate);
			points.y.push_back(coordinate);
			points.z.push_back(coordinate);
		}
		for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		{
			SCOPED_TRACE(::testing::Message() << "'radius' = " << radius << ", 'storage' = " << static_cast<int>(storage));
			cellmate::NeighborSearch search(radius, storage);
			search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), cellmate::ComputeMode::seq);
			std::vector<std::size_t> neighbors, expected;
			for (float coordinate : points.x)
			{
				for (float query_x : { coordinate, std::nextafter(coordinate, -std::numeric_limits<float>::infinity()),
					std::nextafter(coordinate, std::numeric_limits<float>::infinity()) })
				{
					const std::array<float, 3> query_pos{ query_x, coordinate, coordinate };
					search.query(query_pos, neighbors);
					cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(),
						points.x.size(), radius, query_pos, expected);
					std::sort(neighbors.begin(), neighbors.end());
					ASSERT_EQ(neighbors, expected) << "query x = " << query_x;
				}
			}
		}
	}
}

// verifies distant queries remain empty when subtraction overflows in the local frame
TEST(NeighborSearchQuery, DistantFloatQueriesHandleOverflow)
{
	const float far = (std::numeric_limits<float>::max)();
	const float x[]{ far }, yz[1]{};
	const float query_x[]{ -far, far }, query_yz[2]{};
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, yz, yz, 1);
		std::vector<std::size_t> neighbors, offsets;
		search.query({ -far, 0.0f, 0.0f }, neighbors);
		ASSERT_TRUE(neighbors.empty());
		search.query_batch(query_x, query_yz, query_yz, 2, neighbors, offsets);
		ASSERT_EQ(neighbors, (std::vector<std::size_t>{ 0 }));
		ASSERT_EQ(offsets, (std::vector<std::size_t>{ 0, 0, 1 }));
	}
}

// verifies matches and misses when the cell table contains exactly two keys
TEST(NeighborSearchQuery, TwoOccupiedCellsPreserveResults)
{
	const float x[]{ 0.0f, 3.0f }, y[2]{}, z[2]{};
	const float query_x[]{ 3.0f, 1.5f, 0.0f, 3.0f, 1.5f }, query_yz[5]{};
	const std::vector<std::size_t> expected[]{ { 1 }, {}, { 0 }, { 1 }, {} };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage));
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 2, cellmate::ComputeMode::seq);
		std::vector<std::size_t> neighbors, offsets;
		for (std::size_t i = 0; i < 5; i++)
		{
			search.query({ query_x[i], 0.0f, 0.0f }, neighbors);
			ASSERT_EQ(neighbors, expected[i]);
		}
		search.query_batch(query_x, query_yz, query_yz, 5, neighbors, offsets, cellmate::ComputeMode::seq);
		ASSERT_EQ(offsets, (std::vector<std::size_t>{ 0, 1, 1, 2, 3, 3 }));
		ASSERT_EQ(neighbors, (std::vector<std::size_t>{ 1, 0, 1 }));
	}
}

// verifies that queries sharing cell bounds still filter their own positions
TEST(NeighborSearchQuery, RepeatedBoundsPreserveDistanceFiltering)
{
	const float x[]{ 0.0f, 0.2f, 0.6f, 1.2f, 1.8f, 2.0f, 2.9f, 4.0f };
	const float y[8]{}, z[8]{};
	const float query_x[]{ 1.25f, 1.75f, 1.25f, 1.75f };
	const std::vector<std::size_t> expected[]{ { 2, 3, 4, 5 }, { 3, 4, 5 } };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage));
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 8, cellmate::ComputeMode::seq);
		std::vector<std::size_t> neighbors;
		for (std::size_t i = 0; i < 4; i++)
		{
			search.query({ query_x[i], 0.0f, 0.0f }, neighbors);
			ASSERT_EQ(neighbors, expected[i % 2]);
		}
	}
}

// verifies different bounds for queries that share a clamped sorting cell
TEST(NeighborSearchQuery, ClampedQueryCellsPreserveResults)
{
	const float x[]{ 0.0f, 0.5f, 1.5f, 4.0f }, y[4]{}, z[4]{};
	const float query_x[]{ -0.75f, 0.75f, -100.0f, -0.75f, 0.75f };
	const std::vector<std::size_t> expected[]{ { 0 }, { 0, 1, 2 }, {}, { 0 }, { 0, 1, 2 } };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage));
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 4, cellmate::ComputeMode::seq);
		std::vector<std::size_t> neighbors, offsets;
		for (std::size_t i = 0; i < 5; i++)
		{
			search.query({ query_x[i], 0.0f, 0.0f }, neighbors);
			ASSERT_EQ(neighbors, expected[i]);
		}

		// all queries clamp to sorting cell 0, so batches process them in input order
		const float query_yz[5]{};
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			SCOPED_TRACE(::testing::Message() << "'execution' = " << static_cast<int>(execution));
			search.query_batch(query_x, query_yz, query_yz, 5, neighbors, offsets, execution);
			ASSERT_EQ(offsets.size(), 6u);
			ASSERT_EQ(offsets.back(), neighbors.size());
			for (std::size_t i = 0; i < 5; i++)
			{
				const std::vector<std::size_t> row(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
				ASSERT_EQ(row, expected[i]) << "query index = " << i;
			}
		}
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
		std::array<std::thread, 2> threads;
		for (std::size_t thread = 0; thread < threads.size(); thread++)
		{
			threads[thread] = std::thread([&, thread]()
			{
				std::vector<std::size_t> neighbors;
				for (std::size_t i = 0; i < expected.size(); i++)
				{
					search.query({ points.x[i], points.y[i], points.z[i] }, neighbors);
					std::sort(neighbors.begin(), neighbors.end());
					if (neighbors != expected[i]) { valid[thread] = false; }
				}
			});
		}
		for (auto& thread : threads) { thread.join(); }
		for (bool result : valid) { ASSERT_TRUE(result); }
	}
}
