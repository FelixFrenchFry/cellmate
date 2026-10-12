#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <thread>
#include <vector>

// verify that explicit test builds use the requested backend
#if defined(CELLMATE_SIMD_SCALAR)
static_assert(cellmate::detail::DistQuery::N_lanes == 1, "SCALAR tests must use the scalar backend");
#elif defined(CELLMATE_SIMD_AVX256)
static_assert(cellmate::detail::DistQuery::N_lanes == 8, "AVX256 tests must use the AVX256 backend");
#elif defined(CELLMATE_SIMD_AVX512)
static_assert(cellmate::detail::DistQuery::N_lanes == 16, "AVX512 tests must use the AVX512 backend");
#endif

// verifies query-lane masks and the same float cutoff rounding as single-query filtering
TEST(NeighborSearchQueryBatch, QueryGroupsPreserveRoundedCutoffs)
{
	constexpr std::size_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	std::array<float, N_lanes> x{}, y{}, z{};
	unsigned first_mask = 0, second_mask = 0;
	for (std::size_t lane = 0; lane < N_lanes; lane++)
	{
		if (lane % 3 == 0)
		{
			x[lane] = 1.0f;
			y[lane] = std::ldexp(1.0f, -13);
			first_mask |= 1u << lane;
			second_mask |= 1u << lane;
		}
		else if (lane % 3 == 1)
		{
			x[lane] = std::nextafter(1.0f, 2.0f);
		}
		else
		{
			x[lane] = std::ldexp(1.0f, -6);
			y[lane] = 7 * std::ldexp(1.0f, -15);
			z[lane] = 1.0f;
		}
	}
	const cellmate::detail::DistQueryGroup dist_queries(x.data(), y.data(), z.data(), 1.0f);
	EXPECT_EQ(dist_queries.neighbor_mask(0.0f, 0.0f, 0.0f), first_mask);
	EXPECT_EQ(dist_queries.neighbor_mask(0.0f, 0.0f, std::ldexp(1.0f, -13)), second_mask);
	EXPECT_EQ(dist_queries.neighbor_mask(10.0f, 10.0f, 10.0f), 0u);
}

// verifies full query groups, fallback rows, and positions on cell boundaries
// (covers row and neighbor order, both storage modes, and uneven parallel chunks)
TEST(NeighborSearchQueryBatch, HybridGroupsPreserveRows)
{
	cellmate_helpers::PointCloud points{ { 0.0f, 5.0f }, { 0.0f, 5.0f }, { 0.0f, 5.0f } };
	for (int z = 4; z >= 0; z--)
	{
		for (int y = 4; y >= 0; y--)
		{
			for (int x = 4; x >= 0; x--)
			{
				points.x.push_back(float(x) + 0.25f);
				points.y.push_back(float(y) + 0.5f);
				points.z.push_back(float(z) + 0.75f);
			}
		}
	}
	std::vector<std::array<float, 3>> positions;
	for (std::size_t i = 0; i < 33; i++)
	{
		positions.push_back({ 2.25f + float(i % 8) / 16, 2.5f, 2.5f });
	}
	positions.push_back({ 2.0f, 2.0f, 2.0f });
	for (std::size_t i = 0; i < 17; i++)
	{
		positions.push_back({ 2.75f, 2.25f + float(i % 8) / 16, 2.5f });
	}
	const float far = (std::numeric_limits<float>::max)();
	const std::array<float, 3> remaining[]{
		{ -far, 0.0f, 0.0f }, { -0.5f, 0.0f, 0.0f }, { 0.25f, 0.5f, 0.75f },
		{ 1.25f, 0.5f, 0.75f }, { 4.25f, 4.5f, 4.75f }, { far, far, far }, { 5.0f, 5.0f, 5.0f }
	};
	positions.insert(positions.end(), std::begin(remaining), std::end(remaining));
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), cellmate::ComputeMode::seq);
		std::vector<std::vector<std::size_t>> expected(positions.size());
		std::vector<std::size_t> oracle;
		for (std::size_t i = 0; i < positions.size(); i++)
		{
			search.query(positions[i], expected[i]);
			cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), points.x.size(),
				1.0f, positions[i], oracle);
			auto row = expected[i];
			std::sort(row.begin(), row.end());
			ASSERT_EQ(row, oracle);
		}
		std::vector<std::size_t> neighbors, offsets;
		for (std::size_t N_queries : { 7u, 8u, 9u, 15u, 16u, 17u, 33u, 513u, 1031u, 8u })
		{
			cellmate_helpers::PointCloud queries;
			for (std::size_t i = 0; i < N_queries; i++)
			{
				const auto& position = positions[i % positions.size()];
				queries.x.push_back(position[0]);
				queries.y.push_back(position[1]);
				queries.z.push_back(position[2]);
			}
			for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
			{
				for (cellmate::QueryOrder order : { cellmate::QueryOrder::morton, cellmate::QueryOrder::input })
				{
					SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
						<< ", 'execution' = " << static_cast<int>(execution) << ", query order = " << static_cast<int>(order)
						<< ", query count = " << N_queries);
					search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), N_queries, neighbors, offsets,
						execution, order);
					ASSERT_EQ(offsets.size(), N_queries + 1);
					ASSERT_EQ(offsets.front(), 0);
					ASSERT_EQ(offsets.back(), neighbors.size());
					for (std::size_t i = 0; i < N_queries; i++)
					{
						ASSERT_LE(offsets[i], offsets[i + 1]);
						ASSERT_LE(offsets[i + 1], neighbors.size());
						const std::vector<std::size_t> row(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
						ASSERT_EQ(row, expected[i % positions.size()]) << "query index = " << i;
					}
				}
			}
		}
	}
}

// verifies queries and batch rows keep their indices around buffer and SIMD boundaries
TEST(NeighborSearchQueryBatch, DenseRowsPreserveIndicesAcrossBufferBoundaries)
{
	std::vector<float> query_x(513), query_y(513, 0.0f), query_z(513, 0.0f);
	for (std::size_t i = 0; i < query_x.size(); i++) { query_x[i] = i % 2 == 0 ? 0.0f : 10.0f; }
	for (std::size_t N_points : { 0u, 1u, 3u, 4u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u,
		60u, 63u, 64u, 65u, 66u, 67u, 71u, 79u, 127u, 128u, 129u, 257u })
	{
		const std::vector<float> coordinates(N_points, 0.0f);
		std::vector<std::size_t> expected(N_points);
		for (std::size_t i = 0; i < N_points; i++) { expected[i] = i; }
		for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		{
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(coordinates.data(), coordinates.data(), coordinates.data(), N_points, cellmate::ComputeMode::seq);
			std::vector<std::size_t> neighbors{ 99 }, offsets;
			search.query({ 0.0f, 0.0f, 0.0f }, neighbors);
			ASSERT_EQ(neighbors, expected);
			for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
			{
				SCOPED_TRACE(::testing::Message() << "point count = " << N_points
					<< ", 'storage' = " << static_cast<int>(storage) << ", 'execution' = " << static_cast<int>(execution));
				search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(), neighbors, offsets, execution);
				ASSERT_EQ(offsets.size(), query_x.size() + 1);
				std::size_t offset = 0;
				for (std::size_t i = 0; i < query_x.size(); i++)
				{
					ASSERT_EQ(offsets[i], offset);
					if (i % 2 == 0)
					{
						offset += N_points;
						ASSERT_LE(offset, neighbors.size());
						ASSERT_TRUE(std::equal(expected.begin(), expected.end(), neighbors.begin() + offsets[i]));
					}
					ASSERT_EQ(offsets[i + 1], offset);
				}
				ASSERT_EQ(neighbors.size(), offset);
			}
		}
	}
}

// verifies different query masks share tiles without mixing rows or losing buffered indices
TEST(NeighborSearchQueryBatch, HybridPartialMasksPreserveRowsAcrossBufferBoundaries)
{
	std::vector<float> query_x(513), query_y(513), query_z(513);
	for (std::size_t i = 0; i < query_x.size(); i++)
	{
		query_x[i] = float(i % 16) / 16.0f + 0.03125f;
		query_y[i] = float((i * 3) % 16) / 16.0f + 0.03125f;
		query_z[i] = float((i * 7) % 16) / 16.0f + 0.03125f;
	}
	for (std::size_t N_points : { 65u, 71u, 79u, 129u, 257u })
	{
		// all points and queries share one cell, so point indices are already in traversal order
		std::vector<float> x(N_points), y(N_points), z(N_points);
		for (std::size_t i = 0; i < N_points; i++)
		{
			x[i] = float((i * 5) % 16) / 16.0f;
			y[i] = float((i * 7) % 16) / 16.0f;
			z[i] = float((i * 11) % 16) / 16.0f;
		}
		std::vector<std::size_t> expected, expected_offsets{ 0 };
		for (std::size_t query = 0; query < query_x.size(); query++)
		{
			for (std::size_t i = 0; i < N_points; i++)
			{
				const float dx = x[i] - query_x[query];
				const float dy = y[i] - query_y[query];
				const float dz = z[i] - query_z[query];
				if (std::fma(dz, dz, std::fma(dy, dy, dx * dx)) <= 1.0f) { expected.push_back(i); }
			}
			expected_offsets.push_back(expected.size());
		}
		for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		{
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(x.data(), y.data(), z.data(), N_points, cellmate::ComputeMode::seq);
			for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
			{
				SCOPED_TRACE(::testing::Message() << "point count = " << N_points
					<< ", 'storage' = " << static_cast<int>(storage) << ", 'execution' = " << static_cast<int>(execution));
				std::vector<std::size_t> neighbors{ 99 }, offsets;
				search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(), neighbors, offsets, execution);
				ASSERT_EQ(offsets, expected_offsets);
				ASSERT_EQ(neighbors, expected);
			}
		}
	}
}

// verifies fixed cell neighborhoods, including outside queries sharing clamped sorting cells
TEST(NeighborSearchQueryBatch, FixedCellNeighborhoodsPreserveRows)
{
	const float x[]{ 0.0f, std::nextafter(1.0f, 0.0f), 4.0f };
	const float y[]{ 0.0f, 0.25f, 0.25f }, z[]{ 0.0f, 0.25f, 0.25f };
	const float positions[]{ -0.75f, 0.75f, 2.0f, 2.25f, std::nextafter(2.0f, 0.0f), 4.5f, 5.0f, 5.25f };
	const std::vector<std::size_t> expected[]{ { 0 }, { 0, 1 }, { 1 }, {}, { 1 }, { 2 }, { 2 }, {} };
	std::vector<float> query_x(513), query_y(513, 0.25f), query_z(513, 0.25f);
	for (std::size_t i = 0; i < query_x.size(); i++) { query_x[i] = positions[(i / 17) % 8]; }
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 3, cellmate::ComputeMode::seq);
		std::vector<std::size_t> neighbors, offsets;
		for (std::size_t i = 0; i < 8; i++)
		{
			search.query({ positions[i], 0.25f, 0.25f }, neighbors);
			ASSERT_EQ(neighbors, expected[i]) << "query index = " << i;
		}
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			for (cellmate::QueryOrder order : { cellmate::QueryOrder::morton, cellmate::QueryOrder::input })
			{
				search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(), neighbors, offsets, execution, order);
				ASSERT_EQ(offsets.size(), query_x.size() + 1);
				ASSERT_EQ(offsets.back(), neighbors.size());
				for (std::size_t i = 0; i < query_x.size(); i++)
				{
					const std::vector<std::size_t> row(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
					ASSERT_EQ(row, expected[(i / 17) % 8]) << "query index = " << i;
				}
			}
		}
	}
}

// verifies consistent cutoff rounding across scalar tails and complete SIMD blocks
// (covers both storage modes and batch execution)
TEST(NeighborSearchQueryBatch, StorageModesMatchAtRoundedRadiusBoundary)
{
	const float x[17]{}, y[17]{}, z[17]{};
	const std::array<float, 3> query_pos{ 1.0f, std::ldexp(1.0f, -13), 0.0f };
	const std::vector<float> query_x(513, query_pos[0]), query_y(513, query_pos[1]), query_z(513, query_pos[2]);
	for (std::size_t N_points = 0; N_points <= 17; N_points++)
	{
		std::vector<std::size_t> expected(N_points);
		for (std::size_t i = 0; i < N_points; i++) { expected[i] = i; }
		for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		{
			for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
			{
				SCOPED_TRACE(::testing::Message() << "point count = " << N_points
					<< ", 'storage' = " << static_cast<int>(storage) << ", 'execution' = " << static_cast<int>(execution));
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

// verifies the fused z square rounds the distance above the inclusive cutoff
// (covers complete SIMD blocks and a scalar tail, using known expected neighbors)
TEST(NeighborSearchQueryBatch, StorageModesMatchWithRoundedProducts)
{
	// keep inputs runtime values so constant folding cannot hide contraction
	volatile float one = 1.0f;
	const float x[17]{}, y[17]{};
	float z[17];
	std::fill_n(z, 17, std::ldexp(one, -13));
	const std::array<float, 3> query_pos{ std::ldexp(one, -6), 7 * std::ldexp(one, -15), one };
	const std::vector<float> query_x(513, query_pos[0]), query_y(513, query_pos[1]), query_z(513, query_pos[2]);
	const std::vector<std::size_t> expected;
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
				<< ", 'execution' = " << static_cast<int>(execution));
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(x, y, z, 17, execution);
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
			SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
				<< ", 'execution' = " << static_cast<int>(execution));
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

// verifies cell traversal order across contiguous ranges and SIMD boundaries
// (uses reversed input order, empty rows, and exact cell and radius boundaries)
TEST(NeighborSearchQueryBatch, RangesPreserveTraversalOrder)
{
	cellmate_helpers::PointCloud points{ { 0.0f }, { 0.0f }, { 0.0f } };
	for (std::size_t i = 64; i != 0; i--)
	{
		const std::size_t cell = i - 1;
		points.x.push_back(float(cell % 4) + 0.5f);
		points.y.push_back(float((cell / 4) % 4) + 0.5f);
		points.z.push_back(float(cell / 16) + 0.5f);
	}
	const float query_pos[]{ 1.5f, 2.0f, -100.0f };
	const std::vector<std::size_t> expected[]{
		{ 59, 47, 44, 43, 42, 39, 27 }, { 43, 42, 39, 38, 27, 26, 23, 22 }, {}
	};
	std::vector<float> query_x(513), query_y(513), query_z(513);
	for (std::size_t i = 0; i < query_x.size(); i++)
	{
		query_x[i] = query_y[i] = query_z[i] = query_pos[i % 3];
	}
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
				<< ", 'execution' = " << static_cast<int>(execution));
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), execution);
			std::vector<std::size_t> neighbors, offsets, single;
			search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(), neighbors, offsets, execution);
			ASSERT_EQ(offsets.size(), query_x.size() + 1);
			ASSERT_EQ(offsets.front(), 0);
			ASSERT_EQ(offsets.back(), neighbors.size());
			for (std::size_t i = 0; i < query_x.size(); i++)
			{
				ASSERT_EQ(offsets[i + 1] - offsets[i], expected[i % 3].size());
				ASSERT_TRUE(std::equal(expected[i % 3].begin(), expected[i % 3].end(), neighbors.begin() + offsets[i]));
				search.query({ query_x[i], query_y[i], query_z[i] }, single);
				ASSERT_EQ(single, expected[i % 3]);
			}
		}
	}
}

// verifies traversal order as query bounds shift, expand, shrink, and miss the grid
// (covers all adjacent directions, empty cells, and both storage and execution modes)
TEST(NeighborSearchQueryBatch, OverlappingBoundsPreserveTraversalOrder)
{
	cellmate_helpers::PointCloud points{ { 0.0f, 6.0f }, { 0.0f, 6.0f }, { 0.0f, 6.0f } };
	for (int z = 5; z >= 0; z--)
	{
		for (int y = 5; y >= 0; y--)
		{
			for (int x = 5; x >= 0; x--)
			{
				if ((x + 2 * y + 3 * z) % 4 == 0) { continue; }
				for (float offset : { 0.75f, 0.5f, 0.25f })
				{
					points.x.push_back(float(x) + offset);
					points.y.push_back(float(y) + offset);
					points.z.push_back(float(z) + offset);
				}
			}
		}
	}
	std::vector<std::array<float, 3>> positions;
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				positions.push_back({ 2.5f, 2.5f, 2.5f });
				positions.push_back({ 2.5f + float(x), 2.5f + float(y), 2.5f + float(z) });
			}
		}
	}
	const std::array<float, 3> boundaries[]{
		{ 2.0f, 2.0f, 2.0f }, { 2.5f, 2.5f, 2.5f }, { 0.25f, 0.25f, 0.25f },
		{ -0.75f, 0.25f, 0.25f }, { 0.25f, 0.25f, 0.25f }, { 0.25f, 1.25f, 0.25f },
		{ 0.25f, 0.25f, 1.25f }, { 6.25f, 6.25f, 6.25f }, { 5.75f, 5.75f, 5.75f },
		{ 100.0f, 100.0f, 100.0f }, { 3.5f, 3.5f, 3.5f },
		{ 2.0f, 2.5f, 2.5f }, { 2.5f, 2.0f, 2.5f }, { 2.5f, 2.5f, 2.0f }
	};
	positions.insert(positions.end(), std::begin(boundaries), std::end(boundaries));
	std::vector<std::vector<std::size_t>> expected(positions.size());
	for (std::size_t i = 0; i < positions.size(); i++)
	{
		cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), points.x.size(),
			1.0f, positions[i], expected[i]);
		// order the oracle by cell traversal, retaining input order within each cell
		const auto cell = [](float coordinate) { return int(std::floor(double(coordinate) / (1.0 + std::ldexp(1.0, -20)))); };
		std::sort(expected[i].begin(), expected[i].end(), [&](std::size_t a, std::size_t b)
		{
			const std::array<int, 3> first{ cell(points.z[a]), cell(points.y[a]), cell(points.x[a]) };
			const std::array<int, 3> second{ cell(points.z[b]), cell(points.y[b]), cell(points.x[b]) };
			return first != second ? first < second : a < b;
		});
	}
	std::vector<float> query_x(513), query_y(513), query_z(513);
	for (std::size_t i = 0; i < query_x.size(); i++)
	{
		const auto& position = positions[i % positions.size()];
		query_x[i] = position[0];
		query_y[i] = position[1];
		query_z[i] = position[2];
	}
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
				<< ", 'execution' = " << static_cast<int>(execution));
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), execution);
			std::vector<std::size_t> neighbors, offsets;
			for (std::size_t i = 0; i < positions.size(); i++)
			{
				search.query(positions[i], neighbors);
				ASSERT_EQ(neighbors, expected[i]) << "query index = " << i;
			}
			search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(), neighbors, offsets, execution);
			ASSERT_EQ(offsets.size(), query_x.size() + 1);
			ASSERT_EQ(offsets.back(), neighbors.size());
			for (std::size_t i = 0; i < query_x.size(); i++)
			{
				const std::vector<std::size_t> row(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
				ASSERT_EQ(row, expected[i % positions.size()]) << "query index = " << i;
			}
		}
	}
}

// verifies original row order across all Morton key bytes and out-of-grid queries
// (covers repeated keys, empty rows, uneven chunks, and both storage and execution modes)
TEST(NeighborSearchQueryBatch, PreservesRowsAcrossWideExtents)
{
	const float last = float((1u << 21) - 1);
	const float far = (std::numeric_limits<float>::max)();
	const float x[]{ 0.0f, 0.0f, 0.5f, 256.0f, 65536.0f, last };
	const float y[]{ 0.0f, 0.0f, 0.0f, 256.0f, 65536.0f, last };
	const float z[]{ 0.0f, 0.0f, 0.0f, 256.0f, 65536.0f, last };
	const std::array<float, 3> positions[]{
		{ last, last, last }, { -far, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f },
		{ 65536.0f, 65536.0f, 65536.0f }, { far, far, far },
		{ -0.5f, 0.0f, 0.0f }, { 256.0f, 256.0f, 256.0f }, { 10.0f, 10.0f, 10.0f }
	};
	const std::vector<std::size_t> expected[]{ { 5 }, {}, { 0, 1, 2 }, { 4 }, {}, { 0, 1, 2 }, { 3 }, {} };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 6, cellmate::ComputeMode::seq);
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			for (std::size_t N_queries : { 1u, 513u, 8193u })
			{
				SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
					<< ", 'execution' = " << static_cast<int>(execution) << ", query count = " << N_queries);
				cellmate_helpers::PointCloud queries;
				for (std::size_t i = 0; i < N_queries; i++)
				{
					const auto& position = positions[i % 8];
					queries.x.push_back(position[0]);
					queries.y.push_back(position[1]);
					queries.z.push_back(position[2]);
				}
				const auto original = queries;
				std::vector<std::size_t> neighbors{ 99 }, offsets{ 99 };
				search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), N_queries,
					neighbors, offsets, execution);
				ASSERT_EQ(queries.x, original.x);
				ASSERT_EQ(queries.y, original.y);
				ASSERT_EQ(queries.z, original.z);
				ASSERT_EQ(offsets.size(), N_queries + 1);
				std::size_t offset = 0;
				for (std::size_t i = 0; i < N_queries; i++)
				{
					ASSERT_EQ(offsets[i], offset);
					offset += expected[i % 8].size();
					ASSERT_EQ(offsets[i + 1], offset);
					ASSERT_LE(offset, neighbors.size());
					ASSERT_TRUE(std::equal(expected[i % 8].begin(), expected[i % 8].end(), neighbors.begin() + offsets[i]));
				}
				ASSERT_EQ(neighbors.size(), offset);
			}
		}
	}
}

// verifies reused scratch after shrinking batches, switching searches, and rejected inputs
TEST(NeighborSearchQueryBatch, RepeatedBatchesPreserveRows)
{
	const float x[]{ 0.0f, 0.5f, 4.0f }, y[]{ 0.0f, 0.0f, 0.0f }, z[]{ 0.0f, 0.0f, 0.0f };
	std::vector<float> query_x(8193), query_y(8193, 0.0f), query_z(8193, 0.0f);
	for (std::size_t i = 0; i < query_x.size(); i++) { query_x[i] = float((i % 3) * 4); }
	std::vector<std::size_t> neighbors, offsets, expected;
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (std::size_t N_points : { 3u, 1u })
		{
			cellmate::NeighborSearch search(1.0f, storage);
			search.rebuild(x, y, z, N_points, cellmate::ComputeMode::seq);
			for (std::size_t N_queries : { 8193u, 0u, 1u, 513u, 8193u })
			{
				for (cellmate::ComputeMode execution : { cellmate::ComputeMode::par, cellmate::ComputeMode::seq })
				{
					SCOPED_TRACE(::testing::Message() << "point count = " << N_points << ", query count = " << N_queries
						<< ", 'storage' = " << static_cast<int>(storage) << ", 'execution' = " << static_cast<int>(execution));
					search.query_batch(query_x.data(), query_y.data(), query_z.data(), N_queries, neighbors, offsets, execution);
					ASSERT_EQ(offsets.size(), N_queries + 1);
					ASSERT_EQ(offsets.front(), 0);
					ASSERT_EQ(offsets.back(), neighbors.size());
					for (std::size_t i = 0; i < N_queries; i++)
					{
						search.query({ query_x[i], query_y[i], query_z[i] }, expected);
						ASSERT_LE(offsets[i], offsets[i + 1]);
						ASSERT_LE(offsets[i + 1], neighbors.size());
						ASSERT_EQ(offsets[i + 1] - offsets[i], expected.size());
						ASSERT_TRUE(std::equal(expected.begin(), expected.end(), neighbors.begin() + offsets[i]));
					}
				}

				query_x[0] = (std::numeric_limits<float>::quiet_NaN)();
				EXPECT_THROW(search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(),
					neighbors, offsets), std::invalid_argument);
				query_x[0] = 0.0f;
			}

			search.rebuild(nullptr, nullptr, nullptr, 0);
			search.query_batch(query_x.data(), query_y.data(), query_z.data(), query_x.size(), neighbors, offsets);
			EXPECT_TRUE(neighbors.empty());
			EXPECT_EQ(offsets, std::vector<std::size_t>(query_x.size() + 1, 0));
		}
	}
}

// verifies both query orders across reused batches and uneven chunks, including empty chunks
TEST(NeighborSearchQueryBatch, QueryOrdersPreserveRowsAcrossRepeatedBatches)
{
	const float x[]{ 2.0f, 0.0f, 1.0f, 0.0f, 4.0f };
	const float y[5]{}, z[5]{};
	const float positions[]{ 2.0f, 100.0f, 0.0f, 1.0f, -1.0f, 4.0f };
	const std::vector<std::size_t> expected[]{ { 2, 0 }, {}, { 1, 2, 3 }, { 1, 2, 3, 0 }, { 1, 3 }, { 4 } };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 5, cellmate::ComputeMode::seq);
		std::vector<std::size_t> neighbors{ 99 }, offsets{ 99 };
		for (std::size_t N_queries : { 8193u, 1u, 0u, 513u, 8193u })
		{
			std::vector<float> query_x(N_queries), query_y(N_queries, 0.0f), query_z(N_queries, 0.0f);
			for (std::size_t i = 0; i < N_queries; i++)
			{
				query_x[i] = i < N_queries / 2 ? 100.0f : positions[i % 6];
			}
			for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
			{
				for (cellmate::QueryOrder order : { cellmate::QueryOrder::input, cellmate::QueryOrder::morton, cellmate::QueryOrder::input })
				{
					SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
						<< ", 'execution' = " << static_cast<int>(execution) << ", query order = " << static_cast<int>(order)
						<< ", query count = " << N_queries);
					search.query_batch(query_x.data(), query_y.data(), query_z.data(), N_queries, neighbors, offsets, execution, order);
					ASSERT_EQ(offsets.size(), N_queries + 1);
					std::size_t offset = 0;
					for (std::size_t i = 0; i < N_queries; i++)
					{
						const auto& row = expected[i < N_queries / 2 ? 1 : i % 6];
						ASSERT_EQ(query_x[i], i < N_queries / 2 ? 100.0f : positions[i % 6]);
						ASSERT_EQ(query_y[i], 0.0f);
						ASSERT_EQ(query_z[i], 0.0f);
						ASSERT_EQ(offsets[i], offset);
						offset += row.size();
						ASSERT_EQ(offsets[i + 1], offset);
						ASSERT_LE(offset, neighbors.size());
						ASSERT_TRUE(std::equal(row.begin(), row.end(), neighbors.begin() + offsets[i]));
					}
					ASSERT_EQ(offsets.back(), neighbors.size());
					ASSERT_EQ(neighbors.size(), offset);
				}
			}
		}
	}
}

// verifies public validation and empty results when query ordering is bypassed
TEST(NeighborSearchQueryBatch, InputOrderValidatesArgumentsAndHandlesEmptySearches)
{
	cellmate::NeighborSearch search(1.0f);
	const float coordinates[]{ 0.0f };
	const float invalid[]{ (std::numeric_limits<float>::quiet_NaN)() };
	std::vector<std::size_t> neighbors{ 99 }, offsets{ 99 };
	for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
	{
		EXPECT_THROW(search.query_batch(nullptr, coordinates, coordinates, 1, neighbors, offsets,
			execution, cellmate::QueryOrder::input), std::invalid_argument);
		EXPECT_THROW(search.query_batch(coordinates, nullptr, coordinates, 1, neighbors, offsets,
			execution, cellmate::QueryOrder::input), std::invalid_argument);
		EXPECT_THROW(search.query_batch(coordinates, coordinates, nullptr, 1, neighbors, offsets,
			execution, cellmate::QueryOrder::input), std::invalid_argument);
		EXPECT_THROW(search.query_batch(invalid, coordinates, coordinates, 1, neighbors, offsets,
			execution, cellmate::QueryOrder::input), std::invalid_argument);
		EXPECT_THROW(search.query_batch(coordinates, invalid, coordinates, 1, neighbors, offsets,
			execution, cellmate::QueryOrder::input), std::invalid_argument);
		EXPECT_THROW(search.query_batch(coordinates, coordinates, invalid, 1, neighbors, offsets,
			execution, cellmate::QueryOrder::input), std::invalid_argument);
		EXPECT_THROW(search.query_batch(nullptr, nullptr, nullptr, 0, neighbors, neighbors,
			execution, cellmate::QueryOrder::input), std::invalid_argument);
		EXPECT_THROW(search.query_batch(nullptr, nullptr, nullptr, (std::numeric_limits<std::size_t>::max)(), neighbors, offsets,
			execution, cellmate::QueryOrder::input), std::length_error);
		EXPECT_THROW(search.query_batch(nullptr, nullptr, nullptr, 0, neighbors, offsets,
			execution, static_cast<cellmate::QueryOrder>(2)), std::invalid_argument);
		search.query_batch(nullptr, nullptr, nullptr, 0, neighbors, offsets, execution, cellmate::QueryOrder::input);
		EXPECT_TRUE(neighbors.empty());
		EXPECT_EQ(offsets, std::vector<std::size_t>({ 0 }));
		search.query_batch(coordinates, coordinates, coordinates, 1, neighbors, offsets, execution, cellmate::QueryOrder::input);
		EXPECT_TRUE(neighbors.empty());
		EXPECT_EQ(offsets, std::vector<std::size_t>({ 0, 0 }));
	}

	search.rebuild(coordinates, coordinates, coordinates, 1, cellmate::ComputeMode::seq);
	const std::vector<float> outside(513, 100.0f);
	for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
	{
		neighbors.assign(1, 99);
		search.query_batch(outside.data(), outside.data(), outside.data(), outside.size(),
			neighbors, offsets, execution, cellmate::QueryOrder::input);
		EXPECT_TRUE(neighbors.empty());
		EXPECT_EQ(offsets, std::vector<std::size_t>(outside.size() + 1, 0));
	}
}

// verifies that simultaneous batch calls keep their reused sorting and result buffers separate
TEST(NeighborSearchQueryBatch, ConcurrentBatchesMatchBruteForce)
{
	const auto points = cellmate_helpers::make_uniform_cloud(257, 91, -3.0f, 3.0f);
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), cellmate::ComputeMode::seq);
		std::array<bool, 2> valid{ true, true };
		std::array<std::thread, 2> threads;
		for (std::size_t thread = 0; thread < threads.size(); thread++)
		{
			threads[thread] = std::thread([&, thread]()
			{
				const auto queries = cellmate_helpers::make_uniform_cloud(8193, static_cast<unsigned>(67 + thread), -4.0f, 4.0f);
				std::vector<std::size_t> neighbors, offsets, expected;
				const auto order = thread == 0 ? cellmate::QueryOrder::input : cellmate::QueryOrder::morton;
				for (int repeat = 0; repeat < 2; repeat++)
				{
					search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), queries.x.size(),
						neighbors, offsets, cellmate::ComputeMode::par, order);
				}
				if (offsets.size() != queries.x.size() + 1 || offsets.front() != 0 || offsets.back() != neighbors.size())
				{
					valid[thread] = false;
					return;
				}
				for (std::size_t i = 0; i < queries.x.size(); i++)
				{
					cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), points.x.size(),
						1.0f, { queries.x[i], queries.y[i], queries.z[i] }, expected);
					if (offsets[i] > offsets[i + 1] || offsets[i + 1] > neighbors.size())
					{
						valid[thread] = false;
						return;
					}
					std::vector<std::size_t> row(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
					std::sort(row.begin(), row.end());
					if (row != expected) { valid[thread] = false; }
				}
			});
		}
		for (auto& thread : threads) { thread.join(); }
		for (bool result : valid) { ASSERT_TRUE(result); }
	}
}
