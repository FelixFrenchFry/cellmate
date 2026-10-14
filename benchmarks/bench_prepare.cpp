#include "bench_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace cellmate_bench
{
	namespace
	{
		cellmate_helpers::PointCloud make_queries(const cellmate_helpers::PointCloud& points,
			const std::array<float, 3>& half_extent, const Config& config)
		{
			const std::uint32_t seed = config.rng_seed ^ 0x9e3779b9u;
			if (config.query_placement == QueryPlacement::domain)
			{
				auto queries = cellmate_helpers::make_uniform_cloud(config.N_queries, seed, 0.0f, 1.0f);
				std::size_t axis = 0;
				for (auto* coordinates : { &queries.x, &queries.y, &queries.z })
				{
					const float half_side = half_extent[axis++];
					if (!std::isfinite(half_side) || half_side < 0)
					{
						throw std::runtime_error("query domain extents must be finite and nonnegative");
					}
					const float last = std::nextafter(half_side, -half_side);
					for (float& value : *coordinates)
					{
						value = half_side == 0
							? 0.0f
							: (std::min)(last, float((double(value) * 2 - 1) * half_side));
					}
				}
				return queries;
			}
			if (config.N_queries == config.N_points) { return points; }

			cellmate_helpers::PointCloud queries;
			queries.x.resize(config.N_queries);
			queries.y.resize(config.N_queries);
			queries.z.resize(config.N_queries);
			std::mt19937 random(seed);
			for (std::size_t i = 0; i < config.N_queries; i++)
			{
				const std::size_t idx = cellmate_helpers::random_index(random, config.N_points);
				queries.x[i] = points.x[idx];
				queries.y[i] = points.y[idx];
				queries.z[i] = points.z[idx];
			}
			return queries;
		}

		// returns the search radius at which the calibration queries achieve the targeted average neighbor count
		// (choosing the closest achievable count with ties, midway between distinct distances so none lies on the cutoff)
		double calibrate_radius(const cellmate_helpers::PointCloud& points,
			const std::array<float, 3>& half_extent, const Config& config)
		{
			// sample independently of the query count
			Config calibration = config;
			calibration.N_queries = calibration_query_count(config.N_target_avg_nbrs);
			const auto queries = make_queries(points, half_extent, calibration);

			// queries placed on points also find themselves
			const std::size_t N_self = config.query_placement == QueryPlacement::points ? calibration.N_queries : 0;
			const std::size_t N_target = std::size_t(std::llround(config.N_target_avg_nbrs * calibration.N_queries)) + N_self;

			// start slightly above the estimated scale and double until target is reached
			std::vector<std::uint32_t> neighbors;
			std::vector<std::size_t> offsets;
			std::vector<double> distances_squ;
			for (double radius = 1.25 * search_radius;; radius *= 2)
			{
				// find neighbors within the current radius
				cellmate::NeighborSearch search{ float(radius) };
				search.rebuild(points.x.data(), points.y.data(), points.z.data(), config.N_points);
				search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), calibration.N_queries, neighbors, offsets);

				distances_squ.clear();
				for (std::size_t i = 0; i < calibration.N_queries; i++)
				{
					for (std::size_t j = offsets[i]; j < offsets[i + 1]; j++)
					{
						const double dx = double(queries.x[i]) - points.x[neighbors[j]];
						const double dy = double(queries.y[i]) - points.y[neighbors[j]];
						const double dz = double(queries.z[i]) - points.z[neighbors[j]];
						distances_squ.push_back(dx * dx + dy * dy + dz * dz);
					}
				}

				// include or exclude an entire distance group, choosing the closer count
				if (distances_squ.size() > N_target)
				{
					std::nth_element(distances_squ.begin(), distances_squ.begin() + N_target, distances_squ.end());
					const double boundary = distances_squ[N_target];
					std::size_t N_inside = 0, N_equal = 0;
					double inside = 0, outside = radius * radius;
					for (double distance : distances_squ)
					{
						if (distance < boundary)
						{
							N_inside++;
							inside = (std::max)(inside, distance);
						}
						else if (distance == boundary) { N_equal++; }
						else { outside = (std::min)(outside, distance); }
					}
					if (boundary > 0 && N_target - N_inside <= N_inside + N_equal - N_target)
					{
						return (std::sqrt(inside) + std::sqrt(boundary)) / 2;
					}
					if (outside > boundary)
					{
						return (std::sqrt(boundary) + std::sqrt(outside)) / 2;
					}
				}

				// unreachable targets include every point
				if (distances_squ.size() == calibration.N_queries * config.N_points)
				{
					const double max_distance = std::sqrt(*std::max_element(distances_squ.begin(), distances_squ.end()));
					return (std::max)(double(search_radius), max_distance * (1 + std::ldexp(1.0, -10)));
				}
			}
		}
	}

	Workload prepare_workload(const Scenario& scenario, const Config& config)
	{
		if (config.N_points == 0 || config.N_queries == 0 || config.N_max_threads <= 0)
		{
			throw std::invalid_argument("point count, query count and maximum thread count must be positive");
		}
		if (!std::isfinite(config.N_target_avg_nbrs) || config.N_target_avg_nbrs <= 0)
		{
			throw std::invalid_argument("target average neighbors must be finite and positive");
		}
		if (config.compute_mode != cellmate::ComputeMode::seq && config.compute_mode != cellmate::ComputeMode::par)
		{
			throw std::invalid_argument("compute mode must be sequential or parallel");
		}
		if (config.query_placement != QueryPlacement::points &&
			config.query_placement != QueryPlacement::domain)
		{
			throw std::invalid_argument("query placement must be points or domain");
		}

		Workload workload = scenario.make_workload(config);
		if (workload.points.x.size() != config.N_points ||
			workload.points.y.size() != config.N_points ||
			workload.points.z.size() != config.N_points)
		{
			throw std::runtime_error("scenario generated an inconsistent point count");
		}

		workload.queries = make_queries(workload.points, workload.half_extent, config);

		// scale the workload such that the fixed search radius reaches the targeted average neighbor count
		// (caches the calibrated radius per workload for repeated calls, not thread-safe!)
		using CalibrationKey = std::tuple<const Scenario*, std::size_t, double, QueryPlacement, std::uint32_t>;
		static std::map<CalibrationKey, double> calibrated_radii;
		const CalibrationKey key{ &scenario, config.N_points, config.N_target_avg_nbrs, config.query_placement, config.rng_seed };
		auto cached = calibrated_radii.find(key);
		if (cached == calibrated_radii.end())
		{
			cached = calibrated_radii.emplace(key, calibrate_radius(workload.points, workload.half_extent, config)).first;
		}
		if (!std::isnormal(float(cached->second)))
		{
			throw std::runtime_error("calibrated search radius must be a finite positive normal float");
		}

		const double scale = search_radius / cached->second;
		for (float& half_side : workload.half_extent) { half_side = float(half_side * scale); }
		for (auto* cloud : { &workload.points, &workload.queries })
		for (auto* coordinates : { &cloud->x, &cloud->y, &cloud->z })
		for (float& value : *coordinates) { value = float(value * scale); }
		return workload;
	}
}
