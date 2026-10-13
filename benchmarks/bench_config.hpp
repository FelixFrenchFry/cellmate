#pragma once

#include <cellmate/neighbor_search.hpp>

#include "../helpers/point_cloud.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace cellmate_bench
{
	constexpr double pi = 3.14159265358979323846;
	constexpr float search_radius = 1.0f;

	enum class QueryPlacement
	{
		points,
		domain
	};

	struct Config
	{
		std::size_t N_points = 1'000'000;
		std::size_t N_queries = 1'000'000;
		double N_nominal_avg_nbrs = 32.0; // uniform-distribution estimate (excluding self), actual counts depend on the scenario
		int N_max_threads = 4;
		cellmate::ComputeMode compute_mode = cellmate::ComputeMode::par;
		cellmate::PointAccess point_access = cellmate::PointAccess::cpy;
		cellmate::QueryOrder query_order = cellmate::QueryOrder::morton;
		QueryPlacement query_placement = QueryPlacement::points;
		std::uint32_t rng_seed = 67;
	};

	struct Workload
	{
		cellmate_helpers::PointCloud points, queries;
		std::array<float, 3> half_extent;
	};

	struct Scenario
	{
		const char* name;
		std::function<Workload(const Config&)> make_workload;
	};

	Workload prepare_workload(const Scenario& scenario, const Config& config);

	const Scenario& uniform_volume();
	const Scenario& gradient_volume();
	const Scenario& planar_surface();
}
