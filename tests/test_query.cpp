#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../helpers/brute_force.hpp"
#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
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

	std::vector<std::uint32_t> neighbors;
	std::vector<std::uint32_t> expected;
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
			std::vector<std::uint32_t> neighbors;
			std::vector<std::uint32_t> expected;
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

// verifies cutoff neighbors near cell faces far from the grid origin
TEST(NeighborSearchQuery, CellFaceRoundingMatchesBruteForce)
{
	const float origin = 5.07978821f, radius = 0.150742441f;
	const float infinity = std::numeric_limits<float>::infinity();
	cellmate_helpers::PointCloud points{ { origin }, { 0.0f }, { 0.0f } };
	std::vector<float> query_x;
	for (int cell = 1800; cell < 1880; cell++)
	{
		float face = origin + float(cell) * radius;
		for (int i = 0; i < 4; i++) { face = std::nextafter(face, -infinity); }
		for (int i = 0; i < 9; i++, face = std::nextafter(face, infinity))
		{
			query_x.push_back(face);
			float point = face + radius;
			for (int j = 0; j < 4; j++) { point = std::nextafter(point, -infinity); }
			for (int j = 0; j < 9; j++, point = std::nextafter(point, infinity))
			{
				points.x.push_back(point);
				points.y.push_back(0.0f);
				points.z.push_back(0.0f);
			}
		}
	}
	const std::vector<float> query_yz(query_x.size(), 0.0f);

	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage));
		cellmate::NeighborSearch search(radius, storage);
		search.rebuild(points.x.data(), points.y.data(), points.z.data(), points.x.size(), cellmate::ComputeMode::seq);
		std::vector<std::uint32_t> neighbors;
		std::vector<std::uint32_t> expected;
		std::vector<std::size_t> offsets;
		search.query_batch(query_x.data(), query_yz.data(), query_yz.data(), query_x.size(),
			neighbors, offsets, cellmate::ComputeMode::seq);
		for (std::size_t i = 0; i < query_x.size(); i++)
		{
			const std::array<float, 3> query_pos{ query_x[i], 0.0f, 0.0f };
			cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(),
				points.x.size(), radius, query_pos, expected);
			std::vector<std::uint32_t> row(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
			std::sort(row.begin(), row.end());
			ASSERT_EQ(row, expected) << "batch query x = " << query_x[i];

			search.query(query_pos, row);
			std::sort(row.begin(), row.end());
			ASSERT_EQ(row, expected) << "query x = " << query_x[i];
		}
	}
}

// verifies short point ranges with every remainder length, including ranges ending at the stored array end
// (dense clouds keep the following stored points within the radius, so masked-off lanes must not be reported)
TEST(NeighborSearchQuery, RangeRemaindersMatchBruteForce)
{
	const float radius = 1.0f;
	const std::array<std::array<float, 3>, 4> external_queries{ {
		{ 0.0f, 0.0f, 0.0f }, { -2.5f, -2.5f, -2.5f }, { 2.5f, 1.0f, -1.0f }, { 1.75f, -1.75f, 0.5f } } };
	for (std::size_t N_points = 1; N_points <= 40; N_points++)
	{
		const std::uint32_t seed = 101 + static_cast<std::uint32_t>(N_points);
		const cellmate_helpers::PointCloud points = cellmate_helpers::make_uniform_cloud(N_points, seed, -2.0f, 2.0f);
		std::vector<std::array<float, 3>> queries(external_queries.begin(), external_queries.end());
		for (std::size_t i = 0; i < N_points; i++)
		{
			queries.push_back({ points.x[i], points.y[i], points.z[i] });
		}

		for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		{
			SCOPED_TRACE(::testing::Message() << "'N_points' = " << N_points << ", 'storage' = " << static_cast<int>(storage));
			cellmate::NeighborSearch search(radius, storage);
			search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points, cellmate::ComputeMode::seq);
			std::vector<std::uint32_t> neighbors;
			std::vector<std::uint32_t> expected;
			for (const std::array<float, 3>& query_pos : queries)
			{
				search.query(query_pos, neighbors);
				cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(),
					N_points, radius, query_pos, expected);
				std::sort(neighbors.begin(), neighbors.end());
				ASSERT_EQ(neighbors, expected) << "seed = " << seed << ", query = (" << query_pos[0] << ", "
					<< query_pos[1] << ", " << query_pos[2] << ")";
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
		std::vector<std::uint32_t> neighbors;
		std::vector<std::size_t> offsets;
		search.query({ -far, 0.0f, 0.0f }, neighbors);
		ASSERT_TRUE(neighbors.empty());
		search.query_batch(query_x, query_yz, query_yz, 2, neighbors, offsets);
		ASSERT_EQ(neighbors, (std::vector<std::uint32_t>{ 0 }));
		ASSERT_EQ(offsets, (std::vector<std::size_t>{ 0, 0, 1 }));
	}
}

// verifies matches and misses when the cell table contains exactly two keys
TEST(NeighborSearchQuery, TwoOccupiedCellsPreserveResults)
{
	const float x[]{ 0.0f, 3.0f }, y[2]{}, z[2]{};
	const float query_x[]{ 3.0f, 1.5f, 0.0f, 3.0f, 1.5f }, query_yz[5]{};
	const std::vector<std::uint32_t> expected[]{ { 1 }, {}, { 0 }, { 1 }, {} };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage));
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 2, cellmate::ComputeMode::seq);
		std::vector<std::uint32_t> neighbors;
		std::vector<std::size_t> offsets;
		for (std::size_t i = 0; i < 5; i++)
		{
			search.query({ query_x[i], 0.0f, 0.0f }, neighbors);
			ASSERT_EQ(neighbors, expected[i]);
		}
		search.query_batch(query_x, query_yz, query_yz, 5, neighbors, offsets, cellmate::ComputeMode::seq);
		ASSERT_EQ(offsets, (std::vector<std::size_t>{ 0, 1, 1, 2, 3, 3 }));
		ASSERT_EQ(neighbors, (std::vector<std::uint32_t>{ 1, 0, 1 }));
	}
}

// verifies that queries sharing cell bounds still filter their own positions
TEST(NeighborSearchQuery, RepeatedBoundsPreserveDistanceFiltering)
{
	const float x[]{ 0.0f, 0.2f, 0.6f, 1.2f, 1.8f, 2.0f, 2.9f, 4.0f };
	const float y[8]{}, z[8]{};
	const float query_x[]{ 1.25f, 1.75f, 1.25f, 1.75f };
	const std::vector<std::uint32_t> expected[]{ { 2, 3, 4, 5 }, { 3, 4, 5 } };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage));
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 8, cellmate::ComputeMode::seq);
		std::vector<std::uint32_t> neighbors;
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
	const std::vector<std::uint32_t> expected[]{ { 0 }, { 0, 1, 2 }, {}, { 0 }, { 0, 1, 2 } };
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		SCOPED_TRACE(::testing::Message() << "'storage' = " << static_cast<int>(storage));
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 4, cellmate::ComputeMode::seq);
		std::vector<std::uint32_t> neighbors;
		std::vector<std::size_t> offsets;
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
				const std::vector<std::uint32_t> row(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
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
	std::vector<std::vector<std::uint32_t>> expected(32);
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
				std::vector<std::uint32_t> neighbors;
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

// verifies single and batch queries against brute force on random clouds
// (five layouts with random sizes, radii, and cutoff queries, across all modes, query orders, and SoA and AoS inputs)
TEST(NeighborSearchQuery, RandomCloudsMatchBruteForce)
{
	enum class Layout { uniform, clustered, planar, duplicates, lattice };
	std::uint32_t seed = 101;
	for (Layout layout : { Layout::uniform, Layout::clustered, Layout::planar, Layout::duplicates, Layout::lattice })
	{
		for (int repeat = 0; repeat < 3; repeat++, seed++)
		{
			// using 'mt19937' with fixed mappings across standard libraries, upper 24 bits divided by 2^24
			// (exact fp64 products keep the generated clouds identical with and without FMA contraction)
			std::mt19937 random(seed);
			const auto uniform = [&random](float min, float max)
			{
				return float(double(min) + (double(max) - min) * (double(random() >> 8) / 16777216.0));
			};

			// random sizes and radius, with fewer points for layouts where queries have many neighbors
			const bool dense = layout == Layout::clustered || layout == Layout::duplicates;
			const std::size_t N_points = 1 + cellmate_helpers::random_index(random, dense ? 2000 : 12000);
			const std::size_t N_queries = 1 + cellmate_helpers::random_index(random, 600);
			const float extent = uniform(1.0f, 50.0f);
			const float radius = extent * uniform(0.01f, 0.1f);

			// cluster centers and duplicated positions, and lattice sites spaced at the radius or half of it
			std::vector<std::array<float, 3>> centers(1 + cellmate_helpers::random_index(random, 8));
			for (auto& center : centers) { center = { uniform(0.0f, extent), uniform(0.0f, extent), uniform(0.0f, extent) }; }
			const float spacing = repeat % 2 == 0 ? radius : radius / 2;
			std::size_t side = 1;
			while (side * side * side < N_points) { side++; }

			cellmate_helpers::PointCloud points;
			std::vector<std::array<float, 3>> packed_points(N_points);
			for (std::size_t i = 0; i < N_points; i++)
			{
				std::array<float, 3> position{};
				switch (layout)
				{
				case Layout::uniform:
					position = { uniform(0.0f, extent), uniform(0.0f, extent), uniform(0.0f, extent) };
					break;
				case Layout::clustered:
				{
					const auto& center = centers[cellmate_helpers::random_index(random, centers.size())];
					position = { center[0] + uniform(-radius, radius), center[1] + uniform(-radius, radius), center[2] + uniform(-radius, radius) };
					break;
				}
				case Layout::planar:
					position = { uniform(0.0f, extent), uniform(0.0f, extent), 0.0f };
					break;
				case Layout::duplicates:
					position = centers[cellmate_helpers::random_index(random, centers.size())];
					break;
				case Layout::lattice:
					position = { float(i % side) * spacing, float(i / side % side) * spacing, float(i / side / side) * spacing };
					break;
				}
				points.x.push_back(position[0]);
				points.y.push_back(position[1]);
				points.z.push_back(position[2]);
				packed_points[i] = position;
			}

			// queries at points, on the cutoff of points, and around the cloud, including outside the grid
			float low = points.x[0], high = points.x[0];
			for (const std::vector<float>* axis : { &points.x, &points.y, &points.z })
			{
				low = (std::min)(low, *std::min_element(axis->begin(), axis->end()));
				high = (std::max)(high, *std::max_element(axis->begin(), axis->end()));
			}
			cellmate_helpers::PointCloud queries;
			std::vector<std::array<float, 3>> packed_queries(N_queries);
			std::vector<std::vector<std::uint32_t>> expected(N_queries);
			for (std::size_t i = 0; i < N_queries; i++)
			{
				const std::size_t point = cellmate_helpers::random_index(random, N_points);
				std::array<float, 3> position{ points.x[point], points.y[point], points.z[point] };
				if (i % 3 == 1)
				{
					// separate draws keep the axis and sign order fixed before C++17
					const std::size_t axis = cellmate_helpers::random_index(random, 3);
					const float offset = random() % 2 == 0 ? radius : -radius;
					position[axis] += offset;
				}
				else if (i % 3 == 2)
				{
					position = { uniform(low - 2 * radius, high + 2 * radius), uniform(low - 2 * radius, high + 2 * radius),
						uniform(low - 2 * radius, high + 2 * radius) };
				}
				queries.x.push_back(position[0]);
				queries.y.push_back(position[1]);
				queries.z.push_back(position[2]);
				packed_queries[i] = position;
				cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(), N_points,
					radius, position, expected[i]);
			}

			for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
			{
				for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
				{
					for (bool aos : { false, true })
					{
						SCOPED_TRACE(::testing::Message() << "layout = " << static_cast<int>(layout) << ", seed = " << seed
							<< ", point count = " << N_points << ", query count = " << N_queries << ", 'radius' = " << radius
							<< ", 'storage' = " << static_cast<int>(storage) << ", 'execution' = " << static_cast<int>(execution)
							<< ", AoS = " << aos);
						cellmate::NeighborSearch search(radius, storage);
						if (aos) { search.rebuild(packed_points[0].data(), N_points, sizeof(packed_points[0]), execution); }
						else { search.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points, execution); }

						// single queries match brute force, and keep their order for comparing batch rows
						std::vector<std::vector<std::uint32_t>> rows(N_queries);
						std::vector<std::uint32_t> sorted;
						for (std::size_t i = 0; i < N_queries; i++)
						{
							search.query(packed_queries[i], rows[i]);
							sorted = rows[i];
							std::sort(sorted.begin(), sorted.end());
							ASSERT_EQ(sorted, expected[i]) << "query index = " << i;
						}

						std::vector<std::uint32_t> neighbors;
						std::vector<std::size_t> offsets;
						for (cellmate::QueryOrder order : { cellmate::QueryOrder::morton, cellmate::QueryOrder::input })
						{
							if (aos)
							{
								search.query_batch(packed_queries[0].data(), N_queries, sizeof(packed_queries[0]),
									neighbors, offsets, execution, order);
							}
							else
							{
								search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), N_queries,
									neighbors, offsets, execution, order);
							}
							ASSERT_EQ(offsets.size(), N_queries + 1);
							ASSERT_EQ(offsets.front(), 0u);
							ASSERT_EQ(offsets.back(), neighbors.size());
							for (std::size_t i = 0; i < N_queries; i++)
							{
								ASSERT_LE(offsets[i], offsets[i + 1]);
								ASSERT_TRUE(std::equal(rows[i].begin(), rows[i].end(),
									neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]))
									<< "query index = " << i << ", query order = " << static_cast<int>(order);
							}
						}
					}
				}
			}
		}
	}
}
