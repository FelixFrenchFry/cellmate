#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#if CELLMATE_DETAIL_SIMD_WIDTH != 0
namespace
{
	// unclamped fp64 cell coordinate relative to the grid origin, as 'NeighborSearch' computes it
	double reference_cell(float coordinate, float origin, double cell_width_inv)
	{
		return std::floor((double(coordinate) - double(origin)) * cell_width_inv);
	}

	// interleaves three 21-bit coordinates bit by bit, 'x' at bit 0, 'y' at bit 1 and 'z' at bit 2
	std::uint64_t reference_morton_key(std::uint32_t x, std::uint32_t y, std::uint32_t z)
	{
		std::uint64_t key = 0;
		for (unsigned bit = 0; bit < 21; bit++)
		{
			key |= std::uint64_t((x >> bit) & 1u) << (3 * bit);
			key |= std::uint64_t((y >> bit) & 1u) << (3 * bit + 1);
			key |= std::uint64_t((z >> bit) & 1u) << (3 * bit + 2);
		}
		return key;
	}

	// query key clamped to the grid, with the low bit marking cells outside the grid
	std::uint64_t reference_query_key(std::array<float, 3> query_pos, std::array<float, 3> origin, double cell_width_inv,
		std::array<std::uint32_t, 3> last_cell)
	{
		std::uint32_t cells[3];
		bool outside_grid = false;
		for (std::size_t axis = 0; axis < 3; axis++)
		{
			const double cell = reference_cell(query_pos[axis], origin[axis], cell_width_inv);
			outside_grid = outside_grid || cell < 0.0 || cell > double(last_cell[axis]);
			cells[axis] = static_cast<std::uint32_t>((std::max)(0.0, (std::min)(double(last_cell[axis]), cell)));
		}
		return (reference_morton_key(cells[0], cells[1], cells[2]) << 1) | std::uint64_t(outside_grid);
	}
}

// verifies SIMD point and query keys against a bitwise scalar reference
// (random and face coordinates covering all 21 key bits, and queries near and far outside the grid on every side)
TEST(NeighborCellKeys, SimdKeysMatchScalarReference)
{
	const std::size_t N_lanes = cellmate::detail::N_key_lanes;
	const std::size_t N_positions = 64 * N_lanes;
	const float infinity = std::numeric_limits<float>::infinity();

	// unit cells at the origin, and rounded cell widths at an offset origin
	struct Grid
	{
		std::array<float, 3> origin;
		double cell_width_inv;
		float span; // coordinate extent covering almost all 21-bit cell coordinates
	};
	const double cell_width = 0.37 * (1.0 + std::ldexp(1.0, -20));
	const Grid grids[]{
		{ { 0.0f, 0.0f, 0.0f }, 1.0, float((1u << 21) - 1) },
		{ { -3.25f, 0.0f, 1000.0f }, 1.0 / cell_width, float(0.999 * cell_width * double(1u << 21)) }
	};

	std::mt19937 random(17);
	for (std::size_t g = 0; g < sizeof(grids) / sizeof(grids[0]); g++)
	{
		SCOPED_TRACE(::testing::Message() << "grid = " << g);
		const Grid& grid = grids[g];
		std::uniform_real_distribution<float> offset(0.0f, grid.span);

		// points inside the grid, including its faces
		std::array<std::vector<float>, 3> points;
		std::array<float, 3> maximum{};
		std::array<std::uint32_t, 3> last_cell{};
		for (std::size_t axis = 0; axis < 3; axis++)
		{
			std::vector<float>& coordinates = points[axis];
			coordinates.resize(N_positions);
			for (float& coordinate : coordinates) { coordinate = grid.origin[axis] + offset(random); }
			coordinates[0] = grid.origin[axis];
			coordinates[1] = grid.origin[axis] + grid.span;
			coordinates[2] = std::nextafter(coordinates[1], -infinity);
			if (grid.origin[axis] == 0.0f) { coordinates[3] = -0.0f; }

			maximum[axis] = *std::max_element(coordinates.begin(), coordinates.end());
			const double last = reference_cell(maximum[axis], grid.origin[axis], grid.cell_width_inv);
			ASSERT_LT(last, double(1u << 21));
			last_cell[axis] = static_cast<std::uint32_t>(last);
		}

		std::uint64_t keys[cellmate::detail::N_key_lanes];
		for (std::size_t i = 0; i < N_positions; i += N_lanes)
		{
			cellmate::detail::point_cell_keys(points[0].data() + i, points[1].data() + i, points[2].data() + i,
				grid.origin, grid.cell_width_inv, keys);
			for (std::size_t lane = 0; lane < N_lanes; lane++)
			{
				std::uint32_t cells[3];
				for (std::size_t axis = 0; axis < 3; axis++)
				{
					cells[axis] = static_cast<std::uint32_t>(reference_cell(points[axis][i + lane], grid.origin[axis], grid.cell_width_inv));
				}
				ASSERT_EQ(keys[lane], reference_morton_key(cells[0], cells[1], cells[2])) << "point index = " << i + lane;
			}
		}

		// queries inside, next to, and far outside the grid, chosen per axis
		std::array<std::vector<float>, 3> queries;
		for (std::size_t axis = 0; axis < 3; axis++)
		{
			queries[axis].resize(N_positions);
			for (std::size_t i = 0; i < N_positions; i++)
			{
				float& coordinate = queries[axis][i];
				switch (random() % 6)
				{
				case 0: coordinate = points[axis][i]; break;
				case 1: coordinate = grid.origin[axis] - offset(random); break;
				case 2: coordinate = maximum[axis] + offset(random); break;
				case 3: coordinate = std::nextafter(grid.origin[axis], -infinity); break;
				case 4: coordinate = std::nextafter(maximum[axis], infinity); break;
				default: coordinate = (random() % 2 == 0 ? 3.0e38f : -3.0e38f); break;
				}
			}
		}

		for (std::size_t i = 0; i < N_positions; i += N_lanes)
		{
			cellmate::detail::query_cell_keys(queries[0].data() + i, queries[1].data() + i, queries[2].data() + i,
				grid.origin, grid.cell_width_inv, last_cell, keys);
			for (std::size_t lane = 0; lane < N_lanes; lane++)
			{
				const std::array<float, 3> query_pos{ queries[0][i + lane], queries[1][i + lane], queries[2][i + lane] };
				ASSERT_EQ(keys[lane], reference_query_key(query_pos, grid.origin, grid.cell_width_inv, last_cell))
					<< "query index = " << i + lane;
			}
		}
	}
}

// verifies SIMD bounds and finiteness scans against scalar scans
// (every length up to three SIMD batches plus a remainder, with NaN and infinities at every index and axis)
TEST(NeighborCellKeys, SimdScansMatchScalarScans)
{
	const std::size_t N_lanes = cellmate::detail::DistQuery::N_lanes;
	const float nonfinite[]{ (std::numeric_limits<float>::quiet_NaN)(),
		std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() };
	const std::array<float, 3> initial{ 1.0f, -2.0f, 0.5f };
	std::mt19937 random(23);
	std::uniform_real_distribution<float> coordinate(-4.0f, 4.0f);
	for (std::size_t N = 0; N <= 3 * N_lanes + 3; N++)
	{
		SCOPED_TRACE(::testing::Message() << "count = " << N);
		std::array<std::vector<float>, 3> coordinates;
		for (std::vector<float>& values : coordinates)
		{
			values.resize(N);
			for (float& value : values) { value = coordinate(random); }
		}
		// extreme finite values must not count as non-finite
		if (N > 2)
		{
			coordinates[0][N / 2] = (std::numeric_limits<float>::max)();
			coordinates[1][N / 3] = -(std::numeric_limits<float>::max)();
			coordinates[2][N - 1] = (std::numeric_limits<float>::denorm_min)();
		}

		std::array<float, 3> expected_min = initial, expected_max = initial;
		for (std::size_t axis = 0; axis < 3; axis++)
		{
			for (float value : coordinates[axis])
			{
				expected_min[axis] = (std::min)(expected_min[axis], value);
				expected_max[axis] = (std::max)(expected_max[axis], value);
			}
		}
		std::array<float, 3> minimum = initial, maximum = initial;
		ASSERT_TRUE(cellmate::detail::all_finite(coordinates[0].data(), coordinates[1].data(), coordinates[2].data(), N));
		ASSERT_TRUE(cellmate::detail::expand_bounds(coordinates[0].data(), coordinates[1].data(), coordinates[2].data(), N,
			minimum, maximum));
		EXPECT_EQ(minimum, expected_min);
		EXPECT_EQ(maximum, expected_max);

		// one non-finite coordinate at each index and axis
		for (std::size_t i = 0; i < N; i++)
		{
			for (std::size_t axis = 0; axis < 3; axis++)
			{
				for (float value : nonfinite)
				{
					SCOPED_TRACE(::testing::Message() << "index = " << i << ", axis = " << axis << ", value = " << value);
					const float original = coordinates[axis][i];
					coordinates[axis][i] = value;
					minimum = initial;
					maximum = initial;
					EXPECT_FALSE(cellmate::detail::all_finite(coordinates[0].data(), coordinates[1].data(), coordinates[2].data(), N));
					EXPECT_FALSE(cellmate::detail::expand_bounds(coordinates[0].data(), coordinates[1].data(), coordinates[2].data(), N,
						minimum, maximum));
					coordinates[axis][i] = original;
				}
			}
		}
	}
}
#endif

// verifies that non-finite coordinates fail rebuilds and batches at any index and axis
// (in SIMD batches and scalar remainders, for both execution modes and enough points for parallel scans)
TEST(NeighborSearchRebuild, NonFiniteCoordinatesThrowInBatchesAndRemainders)
{
	auto points = cellmate_helpers::make_uniform_cloud(8195, 53, -5.0f, 5.0f);
	float* const coordinates[]{ points.x.data(), points.y.data(), points.z.data() };
	const std::size_t N_points = points.x.size();
	const float nonfinite[]{ (std::numeric_limits<float>::quiet_NaN)(),
		std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() };
	for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
	{
		cellmate::NeighborSearch search(1.0f);
		for (std::size_t i : { std::size_t(0), std::size_t(5), std::size_t(4100), N_points - 3, N_points - 1 })
		{
			for (std::size_t axis = 0; axis < 3; axis++)
			{
				for (float value : nonfinite)
				{
					SCOPED_TRACE(::testing::Message() << "'execution' = " << static_cast<int>(execution)
						<< ", index = " << i << ", axis = " << axis << ", value = " << value);
					const float original = coordinates[axis][i];
					coordinates[axis][i] = value;
					EXPECT_THROW(search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points, execution),
						std::invalid_argument);
					std::vector<std::uint32_t> neighbors;
					std::vector<std::size_t> offsets;
					EXPECT_THROW(search.query_batch(points.x.data(), points.y.data(), points.z.data(), N_points,
						neighbors, offsets, execution), std::invalid_argument);
					coordinates[axis][i] = original;
				}
			}
		}
	}
}
