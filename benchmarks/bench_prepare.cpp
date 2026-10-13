#include "bench_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace cellmate_bench
{
	namespace
	{
		std::size_t random_index(std::mt19937& random, std::size_t N_points)
		{
			const std::uint64_t bound = N_points;
			const std::uint64_t threshold = (std::uint64_t(0) - bound) % bound;
			for (;;)
			{
				const std::uint64_t first = random();
				const std::uint64_t bits = (first << 32) | random();
				if (bits >= threshold)
				{
					return std::size_t(bits % bound);
				}
			}
		}

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
				const std::size_t idx = random_index(random, config.N_points);
				queries.x[i] = points.x[idx];
				queries.y[i] = points.y[idx];
				queries.z[i] = points.z[idx];
			}
			return queries;
		}
	}

	Workload prepare_workload(const Scenario& scenario, const Config& config)
	{
		if (config.N_points == 0 || config.N_queries == 0 || config.N_max_threads <= 0)
		{
			throw std::invalid_argument("point count, query count and maximum thread count must be positive");
		}
		if (!std::isfinite(config.N_nominal_avg_nbrs) || config.N_nominal_avg_nbrs <= 0)
		{
			throw std::invalid_argument("nominal average neighbors must be finite and positive");
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
		return workload;
	}
}
