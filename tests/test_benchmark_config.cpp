#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../benchmarks/bench_config.hpp"
#include "../helpers/brute_force.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

// verifies repeatable generation, geometry and independence from workload settings
TEST(BenchmarkScenarios, CloudsAreDeterministic)
{
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::anisotropic_volume(),
		&cellmate_bench::gradient_volume(), &cellmate_bench::clustered_volume(), &cellmate_bench::lattice_volume(),
		&cellmate_bench::planar_surface(), &cellmate_bench::spherical_surface() })
	{
		SCOPED_TRACE(scenario->name);
		cellmate_bench::Config config;
		config.N_points = 64;
		const auto workload = scenario->make_workload(config);
		const auto& points = workload.points;
		config.N_queries = 97;
		config.N_max_threads = 1;
		config.compute_mode = cellmate::ComputeMode::seq;
		config.point_access = cellmate::PointAccess::ref;
		config.query_order = cellmate::QueryOrder::input;
		config.query_placement = cellmate_bench::QueryPlacement::domain;
		const auto repeated = scenario->make_workload(config).points;
		ASSERT_EQ(points.x.size(), config.N_points);
		ASSERT_EQ(points.y.size(), config.N_points);
		ASSERT_EQ(points.z.size(), config.N_points);
		EXPECT_EQ(points.x, repeated.x);
		EXPECT_EQ(points.y, repeated.y);
		EXPECT_EQ(points.z, repeated.z);

		// original scenarios derive their extents from the reference cube or square, newer ones are checked separately
		const bool planar = scenario == &cellmate_bench::planar_surface();
		const bool spherical = scenario == &cellmate_bench::spherical_surface();
		if (scenario == &cellmate_bench::uniform_volume() || scenario == &cellmate_bench::gradient_volume() ||
			scenario == &cellmate_bench::clustered_volume() || planar)
		{
			const float half_side = planar
				? float(std::sqrt(config.N_points * 3.14159265358979323846 / config.N_target_avg_nbrs)) / 2
				: float(std::cbrt(config.N_points * (4.0 / 3) * 3.14159265358979323846 / config.N_target_avg_nbrs)) / 2;
			EXPECT_FLOAT_EQ(workload.half_extent[0], half_side);
			EXPECT_FLOAT_EQ(workload.half_extent[1], half_side);
			EXPECT_FLOAT_EQ(workload.half_extent[2], planar ? 0.0f : half_side);
		}

		// boxes exclude their upper extent, sphere points may lie on it
		for (std::size_t i = 0; i < config.N_points; i++)
		{
			const std::array<float, 3> position{ points.x[i], points.y[i], points.z[i] };
			for (std::size_t axis = 0; axis < position.size(); axis++)
			{
				const float limit = workload.half_extent[axis];
				EXPECT_GE(position[axis], -limit);
				if (spherical) { EXPECT_LE(position[axis], limit); }
				else if (planar && axis == 2) { EXPECT_FLOAT_EQ(position[axis], 0.0f); }
				else { EXPECT_LT(position[axis], limit); }
			}
		}
		config.rng_seed++;
		EXPECT_NE(points.x, scenario->make_workload(config).points.x);
	}
}

// unequal extents preserve the reference volume and uniform sampling along every axis
TEST(BenchmarkScenarios, AnisotropicVolumePreservesAspectRatio)
{
	cellmate_bench::Config config;
	config.N_points = 10000;
	const auto workload = cellmate_bench::anisotropic_volume().make_workload(config);
	const auto uniform = cellmate_bench::uniform_volume().make_workload(config);
	EXPECT_NEAR(double(workload.half_extent[0]) / workload.half_extent[1], 64, 0.00001);
	EXPECT_FLOAT_EQ(workload.half_extent[1], workload.half_extent[2]);
	double volume_ratio = 1;
	const std::array<const std::vector<float>*, 3> axes{ &workload.points.x, &workload.points.y, &workload.points.z };
	for (std::size_t axis = 0; axis < axes.size(); axis++)
	{
		const double limit = workload.half_extent[axis];
		volume_ratio *= limit / uniform.half_extent[axis];
		ASSERT_EQ(axes[axis]->size(), config.N_points);
		for (float value : *axes[axis])
		{
			EXPECT_GE(value, -limit);
			EXPECT_LT(value, limit);
		}
		for (double fraction : { -0.5, 0.0, 0.5 })
		{
			const auto count = std::count_if(axes[axis]->begin(), axes[axis]->end(),
				[limit, fraction](float value) { return value < limit * fraction; });
			EXPECT_NEAR(double(count) / config.N_points, (fraction + 1) / 2, 0.015);
		}
	}
	EXPECT_NEAR(volume_ratio, 1, 0.000001);
	config.N_points = config.N_queries = 64;
	config.N_target_avg_nbrs = 8;
	for (auto placement : { cellmate_bench::QueryPlacement::points, cellmate_bench::QueryPlacement::domain })
	{
		config.query_placement = placement;
		const auto calibrated = cellmate_bench::prepare_workload(cellmate_bench::anisotropic_volume(), config);
		EXPECT_NEAR(double(calibrated.half_extent[0]) / calibrated.half_extent[1], 64, 0.00001);
		EXPECT_FLOAT_EQ(calibrated.half_extent[1], calibrated.half_extent[2]);
	}
}

// verifies ten equal slabs with linearly increasing density and unchanged uniform y/z coordinates
TEST(BenchmarkScenarios, GradientMatchesDistribution)
{
	cellmate_bench::Config config;
	config.N_points = 10000;
	const auto uniform = cellmate_bench::uniform_volume().make_workload(config).points;
	const auto gradient = cellmate_bench::gradient_volume().make_workload(config).points;
	EXPECT_EQ(gradient.y, uniform.y);
	EXPECT_EQ(gradient.z, uniform.z);
	const double side = std::cbrt(double(config.N_points) * (4.0 / 3) * 3.14159265358979323846 / config.N_target_avg_nbrs);
	double cumulative = 0;
	for (int slab = 0; slab < 10; slab++)
	{
		const double weight = slab + 1;
		// also check halfway through each slab to distinguish steps from a smooth gradient
		const double cutoff = -side / 2 + side * (slab + 0.5) / 10;
		const auto count = std::count_if(gradient.x.begin(), gradient.x.end(),
			[cutoff](float x) { return x < cutoff; });
		const double expected = (cumulative + weight / 2) / 55;
		EXPECT_NEAR(double(count) / config.N_points, expected, 0.015);
		cumulative += weight;
	}
}

// spheres occupy cell centers in a fixed grid and have uniform volume density
TEST(BenchmarkScenarios, ClusteredVolumeUsesRegularGrid)
{
	cellmate_bench::Config config;
	config.N_points = 4096;
	const auto workload = cellmate_bench::clustered_volume().make_workload(config);
	const double half_side = workload.half_extent[0];
	const double cell_side = 2 * half_side / 4;
	const double radius_squared = std::pow(0.45 * cell_side, 2);
	double total_distance_squared = 0;
	for (std::size_t i = 0; i < config.N_points; i++)
	{
		double distance_squared = 0;
		for (float value : { workload.points.x[i], workload.points.y[i], workload.points.z[i] })
		{
			const double cell = std::floor((value + half_side) / cell_side);
			const double offset = value - (-half_side + (cell + 0.5) * cell_side);
			distance_squared += offset * offset;
		}
		ASSERT_LE(distance_squared, radius_squared * 1.000001);
		total_distance_squared += distance_squared;
	}
	EXPECT_NEAR(total_distance_squared / (config.N_points * radius_squared), 3.0 / 5, 0.02);
}

// clustered points form dense pockets while independent domain queries also encounter empty space
TEST(BenchmarkScenarios, ClusteredVolumeHasDensePockets)
{
	cellmate_bench::Config config;
	config.N_points = 32768;
	config.N_queries = 1024;
	config.N_target_avg_nbrs = 2;
	config.query_placement = cellmate_bench::QueryPlacement::domain;
	const auto workload = cellmate_bench::prepare_workload(cellmate_bench::clustered_volume(), config);
	const auto uniform = cellmate_bench::uniform_volume().make_workload(config).points;
	std::size_t clustered_neighbors = 0, uniform_neighbors = 0, empty_queries = 0;
	std::vector<std::uint32_t> neighbors;
	for (std::size_t i = 0; i < 512; i++)
	{
		cellmate_helpers::brute_force(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(),
			config.N_points, cellmate_bench::search_radius,
			{ workload.points.x[i], workload.points.y[i], workload.points.z[i] }, neighbors);
		clustered_neighbors += neighbors.size() - 1;
		cellmate_helpers::brute_force(uniform.x.data(), uniform.y.data(), uniform.z.data(),
			config.N_points, cellmate_bench::search_radius, { uniform.x[i], uniform.y[i], uniform.z[i] }, neighbors);
		uniform_neighbors += neighbors.size() - 1;
	}
	EXPECT_GT(clustered_neighbors, uniform_neighbors * 2);
	for (std::size_t i = 0; i < config.N_queries; i++)
	{
		cellmate_helpers::brute_force(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(),
			config.N_points, cellmate_bench::search_radius,
			{ workload.queries.x[i], workload.queries.y[i], workload.queries.z[i] }, neighbors);
		if (neighbors.empty()) { empty_queries++; }
	}
	EXPECT_GT(empty_queries, config.N_queries / 4);
}

// arbitrary point counts select distinct lattice sites and randomize their storage order
TEST(BenchmarkScenarios, LatticeUsesDistinctShuffledSites)
{
	for (std::size_t N_points : { 1, 2, 8, 10, 27, 64, 100, 125 })
	{
		SCOPED_TRACE(N_points);
		cellmate_bench::Config config;
		config.N_points = N_points;
		const auto workload = cellmate_bench::lattice_volume().make_workload(config);
		std::size_t N_axis = 1;
		while (N_axis * N_axis * N_axis < N_points) { N_axis++; }
		EXPECT_FLOAT_EQ(workload.half_extent[0], float(N_axis) / 2);
		ASSERT_EQ(workload.points.x.size(), N_points);
		ASSERT_EQ(workload.points.y.size(), N_points);
		ASSERT_EQ(workload.points.z.size(), N_points);
		std::set<std::array<float, 3>> sites;
		std::vector<std::size_t> indices;
		for (std::size_t i = 0; i < N_points; i++)
		{
			const std::array<float, 3> site{ workload.points.x[i] + float(N_axis) / 2 - 0.5f,
				workload.points.y[i] + float(N_axis) / 2 - 0.5f,
				workload.points.z[i] + float(N_axis) / 2 - 0.5f };
			for (float value : site)
			{
				EXPECT_FLOAT_EQ(value, std::round(value));
				EXPECT_GE(value, 0);
				EXPECT_LT(value, float(N_axis));
			}
			sites.insert(site);
			indices.push_back(std::size_t(site[0]) + N_axis * (std::size_t(site[1]) + N_axis * std::size_t(site[2])));
		}
		EXPECT_EQ(sites.size(), N_points);
		if (N_points >= 8) { EXPECT_FALSE(std::is_sorted(indices.begin(), indices.end())); }
	}
}

// constant radius and uniform coordinate marginals distinguish area sampling from uniform latitude
TEST(BenchmarkScenarios, SphereSamplesUniformSurfaceArea)
{
	cellmate_bench::Config config;
	config.N_points = 10000;
	const auto workload = cellmate_bench::spherical_surface().make_workload(config);
	const double radius = workload.half_extent[0];
	EXPECT_FLOAT_EQ(float(radius), workload.half_extent[1]);
	EXPECT_FLOAT_EQ(float(radius), workload.half_extent[2]);
	for (std::size_t i = 0; i < config.N_points; i++)
	{
		const double x = workload.points.x[i], y = workload.points.y[i], z = workload.points.z[i];
		EXPECT_NEAR((x * x + y * y + z * z) / (radius * radius), 1, 0.000001);
	}
	for (const auto* coordinates : { &workload.points.x, &workload.points.y, &workload.points.z })
	for (double fraction : { -0.5, 0.0, 0.5 })
	{
		const auto count = std::count_if(coordinates->begin(), coordinates->end(),
			[radius, fraction](float value) { return value < radius * fraction; });
		EXPECT_NEAR(double(count) / config.N_points, (fraction + 1) / 2, 0.015);
	}
}

// calibrated workloads reach the target average neighbors (excluding self) with the fixed search radius
TEST(BenchmarkWorkloads, TargetCountIsReached)
{
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::anisotropic_volume(),
		&cellmate_bench::gradient_volume(), &cellmate_bench::clustered_volume(), &cellmate_bench::lattice_volume(),
		&cellmate_bench::planar_surface(), &cellmate_bench::spherical_surface() })
	for (auto placement : { cellmate_bench::QueryPlacement::points, cellmate_bench::QueryPlacement::domain })
	for (double target : { 0.5, 32.0 })
	{
		// point queries on a lattice have discrete shells, checked separately
		if (scenario == &cellmate_bench::lattice_volume() && placement == cellmate_bench::QueryPlacement::points) { continue; }
		SCOPED_TRACE(scenario->name);
		SCOPED_TRACE(int(placement));
		SCOPED_TRACE(target);
		// query the calibration sample itself (batches match brute force in 'BatchMatchesBruteForce')
		cellmate_bench::Config config;
		config.N_points = 4096;
		config.N_queries = cellmate_bench::calibration_query_count(target);
		config.N_target_avg_nbrs = target;
		config.query_placement = placement;
		const auto workload = cellmate_bench::prepare_workload(*scenario, config);
		cellmate::NeighborSearch search(cellmate_bench::search_radius);
		search.rebuild(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(), config.N_points);
		std::vector<std::uint32_t> neighbors;
		std::vector<std::size_t> offsets;
		search.query_batch(workload.queries.x.data(), workload.queries.y.data(), workload.queries.z.data(),
			config.N_queries, neighbors, offsets);
		const double N_self = placement == cellmate_bench::QueryPlacement::points ? 1 : 0;
		EXPECT_NEAR(double(neighbors.size()) / config.N_queries - N_self, target, 0.005 * target);
	}
}

// tied distances choose the nearer complete shell instead of cutting through it
TEST(BenchmarkWorkloads, TiedDistancesUseNearestCount)
{
	for (const auto& target : { std::array<double, 2>{ 1, 0 }, { 2, 3 }, { 4, 3 }, { 5, 6 } })
	{
		SCOPED_TRACE(target[0]);
		cellmate_bench::Config config;
		config.N_points = config.N_queries = 8;
		config.N_target_avg_nbrs = target[0];
		const auto workload = cellmate_bench::prepare_workload(cellmate_bench::lattice_volume(), config);
		std::vector<std::uint32_t> neighbors;
		for (std::size_t i = 0; i < config.N_queries; i++)
		{
			cellmate_helpers::brute_force(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(),
				config.N_points, cellmate_bench::search_radius,
				{ workload.queries.x[i], workload.queries.y[i], workload.queries.z[i] }, neighbors);
			EXPECT_EQ(double(neighbors.size() - 1), target[1]);
		}
	}
}

// targets beyond the point count include every point instead of shrinking the cloud further
TEST(BenchmarkWorkloads, UnreachableTargetsIncludeAllPoints)
{
	cellmate_bench::Config config;
	config.N_points = config.N_queries = 8;
	config.N_target_avg_nbrs = 32;
	const auto workload = cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config);
	std::vector<std::uint32_t> neighbors;
	for (std::size_t i = 0; i < config.N_queries; i++)
	{
		cellmate_helpers::brute_force(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(),
			config.N_points, cellmate_bench::search_radius,
			{ workload.queries.x[i], workload.queries.y[i], workload.queries.z[i] }, neighbors);
		EXPECT_EQ(neighbors.size(), config.N_points);
	}
}

// verifies repeatable query generation independent of execution settings
TEST(BenchmarkWorkloads, QuerySamplingIsDeterministic)
{
	for (auto placement : { cellmate_bench::QueryPlacement::points, cellmate_bench::QueryPlacement::domain })
	{
		cellmate_bench::Config config;
		config.query_placement = placement;
		config.N_points = 128;
		config.N_queries = 2053;
		config.N_target_avg_nbrs = 8;
		const auto& scenario = cellmate_bench::gradient_volume();
		const auto first = cellmate_bench::prepare_workload(scenario, config);
		config.compute_mode = cellmate::ComputeMode::seq;
		config.point_access = cellmate::PointAccess::ref;
		config.query_order = cellmate::QueryOrder::input;
		config.N_max_threads = 1;
		const auto repeated = cellmate_bench::prepare_workload(scenario, config);
		EXPECT_EQ(first.queries.x, repeated.queries.x);
		EXPECT_EQ(first.queries.y, repeated.queries.y);
		EXPECT_EQ(first.queries.z, repeated.queries.z);
		ASSERT_EQ(first.queries.x.size(), config.N_queries);
		config.N_queries++;
		const auto longer = cellmate_bench::prepare_workload(scenario, config);
		EXPECT_TRUE(std::equal(first.queries.x.begin(), first.queries.x.end(), longer.queries.x.begin()));
		EXPECT_EQ(first.points.x, longer.points.x);
	}
}

// independent queries use the intended domain, including planar geometry and Q=N
TEST(BenchmarkWorkloads, IndependentQueriesUseDomain)
{
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::anisotropic_volume(),
		&cellmate_bench::gradient_volume(), &cellmate_bench::clustered_volume(), &cellmate_bench::lattice_volume(),
		&cellmate_bench::planar_surface(), &cellmate_bench::spherical_surface() })
	{
		SCOPED_TRACE(scenario->name);
		cellmate_bench::Config config;
		config.N_points = config.N_queries = 128;
		config.N_target_avg_nbrs = 8;
		config.query_placement = cellmate_bench::QueryPlacement::domain;
		const auto equal_counts = cellmate_bench::prepare_workload(*scenario, config);
		EXPECT_NE(equal_counts.queries.x, equal_counts.points.x);
		config.N_queries = 4096;
		const auto workload = cellmate_bench::prepare_workload(*scenario, config);
		EXPECT_TRUE(std::equal(equal_counts.queries.x.begin(), equal_counts.queries.x.end(), workload.queries.x.begin()));
		if (scenario == &cellmate_bench::spherical_surface())
		{
			std::size_t N_outside = 0;
			const double radius = workload.half_extent[0];
			for (std::size_t i = 0; i < config.N_queries; i++)
			{
				const double x = workload.queries.x[i], y = workload.queries.y[i], z = workload.queries.z[i];
				if (x * x + y * y + z * z > radius * radius) { N_outside++; }
			}
			EXPECT_NEAR(double(N_outside) / config.N_queries, 1 - cellmate_bench::pi / 6, 0.025);
		}
		const std::array<const std::vector<float>*, 3> axes{ &workload.queries.x, &workload.queries.y, &workload.queries.z };
		for (std::size_t axis = 0; axis < axes.size(); axis++)
		{
			const float limit = workload.half_extent[axis];
			ASSERT_EQ(axes[axis]->size(), config.N_queries);
			for (float value : *axes[axis])
			{
				if (limit == 0) { EXPECT_FLOAT_EQ(value, 0); }
				else
				{
					EXPECT_GE(value, -limit);
					EXPECT_LE(value, limit); // scaling can round onto the bound
				}
			}
			if (limit > 0)
			{
				const auto negative = std::count_if(axes[axis]->begin(), axes[axis]->end(), [](float value) { return value < 0; });
				EXPECT_NEAR(double(negative) / config.N_queries, 0.5, 0.025);
			}
		}
	}
}

// verifies complete batch memberships using the configured execution, storage and ordering modes
TEST(BenchmarkWorkloads, BatchMatchesBruteForce)
{
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::anisotropic_volume(),
		&cellmate_bench::gradient_volume(), &cellmate_bench::clustered_volume(), &cellmate_bench::lattice_volume(),
		&cellmate_bench::planar_surface(), &cellmate_bench::spherical_surface() })
	for (auto placement : { cellmate_bench::QueryPlacement::points, cellmate_bench::QueryPlacement::domain })
	{
		SCOPED_TRACE(scenario->name);
		SCOPED_TRACE(int(placement));
		cellmate_bench::Config config;
		config.N_points = 64;
		config.N_queries = 97;
		config.N_target_avg_nbrs = 8;
		config.query_placement = placement;
		const auto workload = cellmate_bench::prepare_workload(*scenario, config);
		for (auto compute_mode : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		for (auto point_access : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		for (auto query_order : { cellmate::QueryOrder::morton, cellmate::QueryOrder::input })
		{
			cellmate::NeighborSearch search(cellmate_bench::search_radius, point_access);
			search.rebuild(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(),
				config.N_points, compute_mode);
			std::vector<std::uint32_t> neighbors;
			std::vector<std::uint32_t> expected;
			std::vector<std::size_t> offsets;
			search.query_batch(workload.queries.x.data(), workload.queries.y.data(), workload.queries.z.data(),
				config.N_queries, neighbors, offsets, compute_mode, query_order);
			ASSERT_EQ(offsets.size(), config.N_queries + 1);
			ASSERT_EQ(offsets.front(), 0u);
			ASSERT_EQ(offsets.back(), neighbors.size());
			ASSERT_TRUE(std::is_sorted(offsets.begin(), offsets.end()));
			for (std::size_t i = 0; i < config.N_queries; i++)
			{
				cellmate_helpers::brute_force(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(),
					config.N_points, cellmate_bench::search_radius,
					{ workload.queries.x[i], workload.queries.y[i], workload.queries.z[i] }, expected);
				std::vector<std::uint32_t> actual(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
				std::sort(actual.begin(), actual.end());
				EXPECT_EQ(actual, expected) << "query index = " << i;
			}
		}
	}
}

// target counts must be positive, while targets beyond the point count remain valid
TEST(BenchmarkWorkloads, InvalidSettingsAreRejected)
{
	cellmate_bench::Config config;
	config.N_points = config.N_queries = 8;
	for (double target : { 0.0, -1.0, std::numeric_limits<double>::infinity(),
		std::numeric_limits<double>::quiet_NaN() })
	{
		config.N_target_avg_nbrs = target;
		EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	}
	config.N_target_avg_nbrs = 1;
	config.N_queries = 0;
	EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	config.N_queries = 8;
	config.N_max_threads = 0;
	EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	config.N_max_threads = 1;
	config.N_points = 0;
	EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	config.N_points = 1;
	config.N_target_avg_nbrs = 32;
	EXPECT_NO_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config));
	for (int placement : { 2, 99 })
	{
		config.query_placement = static_cast<cellmate_bench::QueryPlacement>(placement);
		EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	}
}
