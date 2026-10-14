#pragma once

#include <cellmate/neighbor_search.hpp>

#include "../helpers/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
		double N_target_avg_nbrs = 32.0; // average neighbors per query (excluding self), calibrated by scaling each workload
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

	// calibration samples about 2^21 neighbors, within bounds for very sparse and dense targets
	inline std::size_t calibration_query_count(double N_target_avg_nbrs)
	{
		return std::size_t((std::min)(std::ldexp(1.0, 20), (std::max)(4096.0, std::round(std::ldexp(1.0, 21) / N_target_avg_nbrs))));
	}

	const Scenario& uniform_volume();
	const Scenario& anisotropic_volume();
	const Scenario& gradient_volume();
	const Scenario& clustered_volume();
	const Scenario& lattice_volume();
	const Scenario& planar_surface();
	const Scenario& spherical_surface();
}
