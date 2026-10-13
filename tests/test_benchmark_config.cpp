#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include "../benchmarks/bench_config.hpp"
#include "../helpers/brute_force.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// verifies repeatable generation, geometry and independence from workload settings
TEST(BenchmarkScenarios, CloudsAreDeterministic)
{
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::gradient_volume(),
		&cellmate_bench::planar_surface() })
	{
		SCOPED_TRACE(scenario->name);
		cellmate_bench::Config config;
		config.N_points = 64;
		const auto points = scenario->make_workload(config).points;
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

		const bool planar = scenario == &cellmate_bench::planar_surface();
		const float half_side = planar
			? float(std::sqrt(config.N_points * 3.14159265358979323846 / config.N_nominal_avg_nbrs)) / 2
			: float(std::cbrt(config.N_points * (4.0 / 3) * 3.14159265358979323846 / config.N_nominal_avg_nbrs)) / 2;
		for (std::size_t i = 0; i < config.N_points; i++)
		{
			EXPECT_GE(points.x[i], -half_side);
			EXPECT_LT(points.x[i], half_side);
			EXPECT_GE(points.y[i], -half_side);
			EXPECT_LT(points.y[i], half_side);
			if (planar) { EXPECT_FLOAT_EQ(points.z[i], 0.0f); }
			else
			{
				EXPECT_GE(points.z[i], -half_side);
				EXPECT_LT(points.z[i], half_side);
			}
		}
		config.rng_seed++;
		EXPECT_NE(points.x, scenario->make_workload(config).points.x);
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
	const double side = std::cbrt(double(config.N_points) * (4.0 / 3) * 3.14159265358979323846 / config.N_nominal_avg_nbrs);
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

// nominal count controls spatial scale directly, while the search radius stays fixed
TEST(BenchmarkWorkloads, NominalCountControlsSpatialScale)
{
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::gradient_volume(),
		&cellmate_bench::planar_surface() })
	{
		SCOPED_TRACE(scenario->name);
		cellmate_bench::Config config;
		config.N_points = config.N_queries = 64;
		const auto reference = cellmate_bench::prepare_workload(*scenario, config);
		const bool planar = scenario == &cellmate_bench::planar_surface();
		for (double nominal : { 0.1, 2.0, 8.0, 32.0, 128.0 })
		{
			SCOPED_TRACE(nominal);
			config.N_nominal_avg_nbrs = nominal;
			const auto workload = cellmate_bench::prepare_workload(*scenario, config);
			EXPECT_EQ(workload.queries.x, workload.points.x);
			EXPECT_EQ(workload.queries.y, workload.points.y);
			EXPECT_EQ(workload.queries.z, workload.points.z);
			const double scale = planar ? std::sqrt(32 / nominal) : std::cbrt(32 / nominal);
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				EXPECT_NEAR(workload.points.x[i], reference.points.x[i] * scale, 0.00001);
				EXPECT_NEAR(workload.points.y[i], reference.points.y[i] * scale, 0.00001);
				EXPECT_NEAR(workload.points.z[i], reference.points.z[i] * scale, 0.00001);
			}
		}
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
		config.N_nominal_avg_nbrs = 8;
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
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::gradient_volume(),
		&cellmate_bench::planar_surface() })
	{
		SCOPED_TRACE(scenario->name);
		cellmate_bench::Config config;
		config.N_points = config.N_queries = 128;
		config.N_nominal_avg_nbrs = 8;
		config.query_placement = cellmate_bench::QueryPlacement::domain;
		const auto equal_counts = cellmate_bench::prepare_workload(*scenario, config);
		EXPECT_NE(equal_counts.queries.x, equal_counts.points.x);
		config.N_queries = 4096;
		const auto workload = cellmate_bench::prepare_workload(*scenario, config);
		EXPECT_TRUE(std::equal(equal_counts.queries.x.begin(), equal_counts.queries.x.end(), workload.queries.x.begin()));
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
					EXPECT_LT(value, limit);
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
	for (const auto* scenario : { &cellmate_bench::uniform_volume(), &cellmate_bench::gradient_volume(),
		&cellmate_bench::planar_surface() })
	for (auto placement : { cellmate_bench::QueryPlacement::points, cellmate_bench::QueryPlacement::domain })
	{
		SCOPED_TRACE(scenario->name);
		SCOPED_TRACE(int(placement));
		cellmate_bench::Config config;
		config.N_points = 64;
		config.N_queries = 97;
		config.N_nominal_avg_nbrs = 8;
		config.query_placement = placement;
		const auto workload = cellmate_bench::prepare_workload(*scenario, config);
		for (auto compute_mode : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
		for (auto point_access : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
		for (auto query_order : { cellmate::QueryOrder::morton, cellmate::QueryOrder::input })
		{
			cellmate::NeighborSearch search(cellmate_bench::search_radius, point_access);
			search.rebuild(workload.points.x.data(), workload.points.y.data(), workload.points.z.data(),
				config.N_points, compute_mode);
			std::vector<std::size_t> neighbors, offsets, expected;
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
				std::vector<std::size_t> actual(neighbors.begin() + offsets[i], neighbors.begin() + offsets[i + 1]);
				std::sort(actual.begin(), actual.end());
				EXPECT_EQ(actual, expected) << "query index = " << i;
			}
		}
	}
}

// nominal counts are positive estimates, not achievable targets limited by the point count
TEST(BenchmarkWorkloads, InvalidSettingsAreRejected)
{
	cellmate_bench::Config config;
	config.N_points = config.N_queries = 8;
	for (double nominal : { 0.0, -1.0, std::numeric_limits<double>::infinity(),
		std::numeric_limits<double>::quiet_NaN() })
	{
		config.N_nominal_avg_nbrs = nominal;
		EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	}
	config.N_nominal_avg_nbrs = 1;
	config.N_queries = 0;
	EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	config.N_queries = 8;
	config.N_max_threads = 0;
	EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	config.N_max_threads = 1;
	config.N_points = 0;
	EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	config.N_points = 1;
	config.N_nominal_avg_nbrs = 32;
	EXPECT_NO_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config));
	for (int placement : { 2, 99 })
	{
		config.query_placement = static_cast<cellmate_bench::QueryPlacement>(placement);
		EXPECT_THROW(cellmate_bench::prepare_workload(cellmate_bench::uniform_volume(), config), std::invalid_argument);
	}
}
