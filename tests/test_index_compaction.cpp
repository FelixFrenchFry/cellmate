#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
	struct PointEntry
	{
		std::uint64_t key;
		std::size_t idx;
	};

	const std::array<PointEntry, 16> entries{
		PointEntry{ 91, static_cast<std::size_t>(0x100000001ull) },
		PointEntry{ 82, static_cast<std::size_t>(0x200000003ull) },
		PointEntry{ 73, static_cast<std::size_t>(0x300000007ull) },
		PointEntry{ 64, static_cast<std::size_t>(0x40000000full) },
		PointEntry{ 55, static_cast<std::size_t>(0x50000001full) },
		PointEntry{ 46, static_cast<std::size_t>(0x60000003full) },
		PointEntry{ 37, static_cast<std::size_t>(0x70000007full) },
		PointEntry{ 28, static_cast<std::size_t>(0x8000000ffull) },
		PointEntry{ 19, static_cast<std::size_t>(0x9000001ffull) },
		PointEntry{ 10, static_cast<std::size_t>(0xa000003ffull) },
		PointEntry{ 99, static_cast<std::size_t>(0xb000007ffull) },
		PointEntry{ 88, static_cast<std::size_t>(0xc00000fffull) },
		PointEntry{ 77, static_cast<std::size_t>(0xd00001fffull) },
		PointEntry{ 66, static_cast<std::size_t>(0xe00003fffull) },
		PointEntry{ 55, static_cast<std::size_t>(0xf00007fffull) },
		PointEntry{ 44, static_cast<std::size_t>(0x100000ffffull) }
	};
}

// verifies every mask preserves full indices and lane order without writing past the output slots
TEST(NeighborIndexCompaction, MasksPreserveIndicesAndOrder)
{
	const std::size_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	const std::size_t guard = (std::numeric_limits<std::size_t>::max)();
	for (unsigned mask = 0; mask < (1u << N_lanes); mask++)
	{
		SCOPED_TRACE(::testing::Message() << "mask = " << mask);
		std::array<std::size_t, cellmate::detail::DistQuery::N_lanes + 2> output;
		output.fill(guard);
		std::vector<std::size_t> expected;
		for (std::size_t lane = 0; lane < N_lanes; lane++)
		{
			if ((mask & (1u << lane)) != 0) { expected.push_back(entries[lane].idx); }
		}
		const std::size_t count = cellmate::detail::pack_neighbor_indices(entries.data(), mask, output.data() + 1);
		ASSERT_EQ(count, expected.size());
		ASSERT_TRUE(std::equal(expected.begin(), expected.end(), output.begin() + 1));
		ASSERT_EQ(output.front(), guard);
		ASSERT_EQ(output.back(), guard);
	}
}

// verifies partial masks crossing the buffer boundary, scalar tails, and appending batch rows
TEST(NeighborSearchQuery, PartialMasksPreserveIndicesAcrossBufferBoundaries)
{
	const std::size_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	for (std::size_t N_prefix : { 64u, 128u })
	{
		for (std::size_t N_rejected : { 0u, 1u, 2u, 3u, 7u })
		{
			for (unsigned pattern : { 0u, 1u, 3u, 0x8u, 0xfu, 0x10u, 0x55u, 0x80u, 0xaau,
				0xffu, 0x100u, 0x5555u, 0x8000u, 0xaaaau, 0xff00u, 0xffffu })
			{
				const unsigned mask = pattern & ((1u << N_lanes) - 1);
				SCOPED_TRACE(::testing::Message() << "prefix = " << N_prefix
					<< ", rejected = " << N_rejected << ", mask = " << mask);
				// all candidates share one cell, (0.75, 0.75, 0.75) lies outside the radius
				std::vector<float> coordinates(N_prefix + N_lanes + 1, 0.0f);
				std::vector<std::size_t> expected;
				for (std::size_t i = 0; i < N_prefix; i++)
				{
					if (i < N_rejected) { coordinates[i] = 0.75f; }
					else { expected.push_back(i); }
				}
				for (std::size_t lane = 0; lane < N_lanes; lane++)
				{
					if ((mask & (1u << lane)) != 0) { expected.push_back(N_prefix + lane); }
					else { coordinates[N_prefix + lane] = 0.75f; }
				}
				expected.push_back(coordinates.size() - 1);
				for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
				{
					cellmate::NeighborSearch search(1.0f, storage);
					search.rebuild(coordinates.data(), coordinates.data(), coordinates.data(), coordinates.size(), cellmate::ComputeMode::seq);
					std::vector<std::size_t> neighbors{ 99 }, offsets;
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
					<< ", storage = " << static_cast<int>(storage) << ", execution = " << static_cast<int>(execution));
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
