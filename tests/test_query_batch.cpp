#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <vector>

// verify that explicit test builds use the requested backend
#if defined(CELLMATE_SIMD_SCALAR)
static_assert(cellmate::detail::DistQuery::N_lanes == 1, "SCALAR tests must use the scalar backend");
#elif defined(CELLMATE_SIMD_AVX256)
static_assert(cellmate::detail::DistQuery::N_lanes == 4, "AVX256 tests must use the AVX256 backend");
#elif defined(CELLMATE_SIMD_AVX512)
static_assert(cellmate::detail::DistQuery::N_lanes == 8, "AVX512 tests must use the AVX512 backend");
#endif

// verifies consistent cutoff rounding across scalar tails and complete SIMD blocks
// (also covers extended scalar precision, both storage modes, and batch execution)
TEST(NeighborSearchQueryBatch, StorageModesMatchAtRoundedRadiusBoundary)
{
	const float x[17]{}, y[17]{}, z[17]{};
	const std::array<float, 3> query_pos{ 1.0f, std::ldexp(1.0f, -27), 0.0f };
	const std::vector<float> query_x(513, query_pos[0]), query_y(513, query_pos[1]), query_z(513, query_pos[2]);
	for (std::size_t N_points = 0; N_points <= 17; N_points++)
	{
		std::vector<std::size_t> expected;
		cellmate_helpers::brute_force(x, y, z, N_points, 1.0f, query_pos, expected);
		for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		{
			for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
			{
				SCOPED_TRACE(::testing::Message() << "point count = " << N_points
					<< ", storage = " << static_cast<int>(storage) << ", execution = " << static_cast<int>(execution));
				cellmate::NeighborSearch search(1.0f, storage);
				search.rebuild(x, y, z, N_points, execution);
				std::vector<std::size_t> neighbors, offsets;
				search.query(query_pos, neighbors);
				ASSERT_EQ(neighbors, expected);
				search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(),
					neighbors, offsets, execution);
				ASSERT_EQ(offsets.size(), query_x.size() + 1);
				ASSERT_EQ(neighbors.size(), query_x.size() * expected.size());
				for (std::size_t i = 0; i < query_x.size(); i++)
				{
					ASSERT_EQ(offsets[i], i * expected.size());
					ASSERT_TRUE(std::equal(expected.begin(), expected.end(), neighbors.begin() + offsets[i]));
				}
				ASSERT_EQ(offsets.back(), neighbors.size());
			}
		}
	}
}

// verifies consistent cutoff rounding when squared distances require rounding
// (covers complete SIMD blocks and a scalar tail, using known expected neighbors)
TEST(NeighborSearchQueryBatch, StorageModesMatchWithRoundedProducts)
{
	if (FLT_EVAL_METHOD != 0 && FLT_EVAL_METHOD != 1)
	{
		GTEST_SKIP() << "requires double evaluation precision";
	}

	// keep inputs runtime values so constant folding cannot hide contraction
	volatile float one = 1.0f;
	const float x[17]{}, y[17]{};
	float z[17];
	std::fill_n(z, 17, std::ldexp(one, -27));
	const std::array<float, 3> query_pos{ std::ldexp(one, -13), 5 * std::ldexp(one, -29), one };
	const std::vector<float> query_x(513, query_pos[0]), query_y(513, query_pos[1]), query_z(513, query_pos[2]);
	std::vector<std::size_t> expected(17);
	for (std::size_t i = 0; i < expected.size(); i++) { expected[i] = i; }
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			SCOPED_TRACE(::testing::Message() << "storage = " << static_cast<int>(storage)
				<< ", execution = " << static_cast<int>(execution));
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(x, y, z, expected.size(), execution);
			std::vector<std::size_t> neighbors, offsets;
			search.query(query_pos, neighbors);
			ASSERT_EQ(neighbors, expected);
			search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(),
				neighbors, offsets, execution);
			ASSERT_EQ(offsets.size(), query_x.size() + 1);
			ASSERT_EQ(neighbors.size(), query_x.size() * expected.size());
			for (std::size_t i = 0; i < query_x.size(); i++)
			{
				ASSERT_EQ(offsets[i], i * expected.size());
				ASSERT_TRUE(std::equal(expected.begin(), expected.end(), neighbors.begin() + offsets[i]));
			}
			ASSERT_EQ(offsets.back(), neighbors.size());
		}
	}
}

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
