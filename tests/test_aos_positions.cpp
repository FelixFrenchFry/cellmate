#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
	// AoS point with extra fields after its coordinates, like padded point types of other libraries
	// (tests store NaN in the extra fields, which rebuilds and queries must never read)
	struct Particle
	{
		float x, y, z;
		float mass;
		std::uint32_t id;
	};
}

// verifies that AoS positions match SoA positions exactly
// (packed, padded and flat layouts across all storage modes, execution modes and query orders)
TEST(NeighborSearchAosPositions, AosInputsMatchSoaInputs)
{
	const auto points = cellmate_helpers::make_uniform_cloud(10003, 23, -5.0f, 5.0f);
	const auto queries = cellmate_helpers::make_uniform_cloud(2003, 29, -2.0f, 2.0f);
	const std::size_t N_points = points.x.size();
	const std::size_t N_queries = queries.x.size();

	// packed (x,y,z), padded particles, and a flat x, y, z buffer of the same points and queries
	const float nan = (std::numeric_limits<float>::quiet_NaN)();
	std::vector<std::array<float, 3>> packed(N_points);
	std::vector<Particle> particles(N_points);
	std::vector<float> flat(3 * N_points);
	for (std::size_t i = 0; i < N_points; i++)
	{
		packed[i] = { points.x[i], points.y[i], points.z[i] };
		particles[i] = { points.x[i], points.y[i], points.z[i], nan, 0 };
		flat[3 * i] = points.x[i];
		flat[3 * i + 1] = points.y[i];
		flat[3 * i + 2] = points.z[i];
	}
	std::vector<Particle> query_particles(N_queries);
	for (std::size_t i = 0; i < N_queries; i++)
	{
		query_particles[i] = { queries.x[i], queries.y[i], queries.z[i], nan, 0 };
	}

	struct AosInput
	{
		const float* xyz;
		std::size_t stride_bytes;
	};
	const AosInput inputs[]{
		{ packed[0].data(), sizeof(packed[0]) },
		{ &particles[0].x, sizeof(Particle) },
		{ flat.data(), 3 * sizeof(float) }
	};

	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		{
			cellmate::NeighborSearch expected_search(1.0f, storage);
			expected_search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points, execution);

			for (std::size_t input = 0; input < std::size(inputs); input++)
			{
				SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage)
					<< ", 'execution' = " << static_cast<int>(execution) << ", 'input' = " << input);
				cellmate::NeighborSearch search(1.0f, storage);
				search.rebuild(inputs[input].xyz, N_points, inputs[input].stride_bytes, execution);

				std::vector<std::uint32_t> single, expected_single;
				for (std::size_t i = 0; i < N_queries; i += 97)
				{
					const std::array<float, 3> query_pos{ queries.x[i], queries.y[i], queries.z[i] };
					search.query(query_pos, single);
					expected_search.query(query_pos, expected_single);
					ASSERT_EQ(single, expected_single) << "query index = " << i;
				}

				for (cellmate::QueryOrder order : { cellmate::QueryOrder::morton, cellmate::QueryOrder::input })
				{
					std::vector<std::uint32_t> neighbors, expected_neighbors;
					std::vector<std::size_t> offsets, expected_offsets;
					search.query_batch(&query_particles[0].x, N_queries, sizeof(Particle), neighbors, offsets, execution, order);
					expected_search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), N_queries,
						expected_neighbors, expected_offsets, execution, order);
					ASSERT_EQ(offsets, expected_offsets);
					ASSERT_EQ(neighbors, expected_neighbors);
				}
			}
		}
	}

	// cross-check one AoS query against brute force
	cellmate::NeighborSearch search(1.0f);
	search.rebuild(&particles[0].x, N_points, sizeof(Particle));
	std::vector<std::uint32_t> neighbors, expected;
	search.query({ 0.0f, 0.0f, 0.0f }, neighbors);
	cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), N_points,
		1.0f, { 0.0f, 0.0f, 0.0f }, expected);
	std::sort(neighbors.begin(), neighbors.end());
	ASSERT_EQ(neighbors, expected);
}

// verifies that invalid AoS strides and missing position arrays are rejected, leaving an empty search
TEST(NeighborSearchAosPositions, RejectsInvalidStridesAndMissingArrays)
{
	const float xyz[8]{};
	std::vector<std::uint32_t> neighbors;
	std::vector<std::size_t> offsets;
	for (std::size_t stride : { std::size_t(0), std::size_t(8), std::size_t(14) })
	{
		SCOPED_TRACE(::testing::Message() << "'stride' = " << stride);
		cellmate::NeighborSearch search(1.0f);
		search.rebuild(xyz, 1, 3 * sizeof(float));
		EXPECT_THROW(search.rebuild(xyz, 1, stride), std::invalid_argument);
		search.query({ 0.0f, 0.0f, 0.0f }, neighbors);
		EXPECT_TRUE(neighbors.empty());
		EXPECT_THROW(search.query_batch(xyz, 1, stride, neighbors, offsets), std::invalid_argument);
	}

	cellmate::NeighborSearch search(1.0f);
	EXPECT_THROW(search.rebuild(nullptr, 1, 3 * sizeof(float)), std::invalid_argument);
	EXPECT_THROW(search.query_batch(nullptr, 1, 3 * sizeof(float), neighbors, offsets), std::invalid_argument);

	// empty inputs need no positions
	search.rebuild(nullptr, 0, 3 * sizeof(float));
	search.query_batch(nullptr, 0, 3 * sizeof(float), neighbors, offsets);
	EXPECT_TRUE(neighbors.empty());
	EXPECT_EQ(offsets, std::vector<std::size_t>({ 0 }));
}
