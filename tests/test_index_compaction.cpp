#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

TEST(NeighborIndexCompaction, GatherCoordinatesClearsInactiveLanes)
{
	struct Point { float x, y, z, padding; };
	constexpr std::size_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	std::array<Point, N_lanes> points{};
	std::array<std::uint32_t, N_lanes> indices{};
	for (std::size_t lane = 0; lane < N_lanes; lane++)
	{
		const float value = static_cast<float>(lane + 1);
		points[lane] = { value, -value, value + 10.0f, 0.0f };
		indices[lane] = static_cast<std::uint32_t>(N_lanes - 1 - lane);
	}
	const std::array<const float*, 3> coords{ &points[0].x, &points[0].y, &points[0].z };
	for (std::size_t N_valid = 0; N_valid <= N_lanes; N_valid++)
	{
		float gathered[3][N_lanes];
		for (auto& axis : gathered) { std::fill_n(axis, N_lanes, -100.0f); }
		cellmate::detail::gather_coords(coords, indices.data(), sizeof(Point) / sizeof(float), N_valid, gathered);
		for (std::size_t lane = 0; lane < N_lanes; lane++)
		{
			SCOPED_TRACE(::testing::Message() << "valid lanes = " << N_valid << ", lane = " << lane);
			const Point expected = lane < N_valid ? points[indices[lane]] : Point{};
			EXPECT_EQ(gathered[0][lane], expected.x);
			EXPECT_EQ(gathered[1][lane], expected.y);
			EXPECT_EQ(gathered[2][lane], expected.z);
		}
	}
}

// verifies every mask preserves full indices and lane order from a contiguous index array
// (without writing past the output slots)
TEST(NeighborIndexCompaction, ContiguousMasksPreserveIndicesAndOrder)
{
	const std::size_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	const std::uint32_t guard = (std::numeric_limits<std::uint32_t>::max)();
	std::array<std::uint32_t, 16> indices;
	for (std::size_t lane = 0; lane < indices.size(); lane++) { indices[lane] = 0x80000000u | ((2u << lane) - 1u); }
	for (unsigned mask = 0; mask < (1u << N_lanes); mask++)
	{
		SCOPED_TRACE(::testing::Message() << "'mask' = " << mask);
		std::array<std::uint32_t, cellmate::detail::DistQuery::N_lanes + 2> output;
		output.fill(guard);
		std::vector<std::uint32_t> expected;
		for (std::size_t lane = 0; lane < N_lanes; lane++)
		{
			if ((mask & (1u << lane)) != 0) { expected.push_back(indices[lane]); }
		}
		const std::size_t count = cellmate::detail::pack_contiguous_indices(indices.data(), mask, output.data() + 1);
		ASSERT_EQ(count, expected.size());
		ASSERT_TRUE(std::equal(expected.begin(), expected.end(), output.begin() + 1));
		ASSERT_EQ(output.front(), guard);
		ASSERT_EQ(output.back(), guard);
	}
}

// verifies every mask preserves float bits and lane order without writing past the output slots
TEST(NeighborIndexCompaction, ContiguousMasksPreserveFloatBitsAndOrder)
{
	constexpr std::size_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	const float guard = (std::numeric_limits<float>::lowest)();
	std::array<float, N_lanes> values;
	for (std::size_t lane = 0; lane < N_lanes; lane++) { values[lane] = -static_cast<float>(lane) * 1.25f; }
	for (unsigned mask = 0; mask < (1u << N_lanes); mask++)
	{
		SCOPED_TRACE(::testing::Message() << "'mask' = " << mask);
		std::array<float, N_lanes + 2> output;
		output.fill(guard);
		std::array<float, N_lanes> expected;
		std::size_t N_selected = 0;
		for (std::size_t lane = 0; lane < N_lanes; lane++)
		{
			if ((mask & (1u << lane)) != 0) { expected[N_selected++] = values[lane]; }
		}
		cellmate::detail::pack_contiguous_floats(values.data(), mask, output.data() + 1);
		ASSERT_EQ(std::memcmp(expected.data(), output.data() + 1, N_selected * sizeof(float)), 0);
		ASSERT_EQ(output.front(), guard);
		ASSERT_EQ(output.back(), guard);
	}
}

// verifies partial masks crossing the buffer boundary, scalar tails, and appending batch rows
TEST(NeighborSearchQuery, PartialMasksPreserveIndicesAcrossBufferBoundaries)
{
	const std::uint32_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	for (std::uint32_t N_prefix : { 64u, 128u })
	{
		for (std::uint32_t N_rejected : { 0u, 1u, 2u, 3u, 7u })
		{
			for (unsigned pattern : { 0u, 1u, 3u, 0x8u, 0xfu, 0x10u, 0x55u, 0x80u, 0xaau,
				0xffu, 0x100u, 0x5555u, 0x8000u, 0xaaaau, 0xff00u, 0xffffu })
			{
				const unsigned mask = pattern & ((1u << N_lanes) - 1);
				SCOPED_TRACE(::testing::Message() << "prefix = " << N_prefix
					<< ", rejected = " << N_rejected << ", 'mask' = " << mask);
				// all candidates share one cell, (0.75, 0.75, 0.75) lies outside the radius
				std::vector<float> coordinates(N_prefix + N_lanes + 1, 0.0f);
				std::vector<std::uint32_t> expected;
				for (std::uint32_t i = 0; i < N_prefix; i++)
				{
					if (i < N_rejected) { coordinates[i] = 0.75f; }
					else { expected.push_back(i); }
				}
				for (std::uint32_t lane = 0; lane < N_lanes; lane++)
				{
					if ((mask & (1u << lane)) != 0) { expected.push_back(N_prefix + lane); }
					else { coordinates[N_prefix + lane] = 0.75f; }
				}
				expected.push_back(N_prefix + N_lanes);
				for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
				{
					cellmate::NeighborSearch search(1.0f, storage);
					search.rebuild(coordinates.data(), coordinates.data(), coordinates.data(), coordinates.size(), cellmate::ComputeMode::seq);
					std::vector<std::uint32_t> neighbors{ 99 };
					std::vector<std::size_t> offsets;
					search.query({ 0.0f, 0.0f, 0.0f }, neighbors);
					ASSERT_EQ(neighbors, expected);
					const float queries[2]{};
					search.query_batch(queries, queries, queries, 2, neighbors, offsets, cellmate::ComputeMode::seq);
					ASSERT_EQ(offsets, (std::vector<std::size_t>{ 0, expected.size(), 2 * expected.size() }));
					ASSERT_EQ(neighbors.size(), 2 * expected.size());
					ASSERT_TRUE(std::equal(expected.begin(), expected.end(), neighbors.begin()));
					ASSERT_TRUE(std::equal(expected.begin(), expected.end(), neighbors.begin() + expected.size()));
				}
			}
		}
	}
}
