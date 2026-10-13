#include "../bench_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace cellmate_bench
{
	namespace
	{
		struct Params
		{
			std::size_t N_clusters_per_axis = 4;
			double cluster_radius_ratio = 0.45; // fraction of the grid cell side
			float half_side = 0.0f;
			float cluster_radius = 0.0f;
		};

		Params derive_params(const Config& config, Params params)
		{
			if (params.N_clusters_per_axis == 0 || !std::isfinite(params.cluster_radius_ratio) ||
				params.cluster_radius_ratio <= 0 || params.cluster_radius_ratio >= 0.5)
			{
				throw std::invalid_argument("clusters per axis must be positive and radius ratio must be between zero and one half");
			}
			params.half_side = float(search_radius * std::cbrt(
				double(config.N_points) * (4.0 / 3) * pi / config.N_target_avg_nbrs)) / 2;
			params.cluster_radius = float(2 * double(params.half_side) * params.cluster_radius_ratio / params.N_clusters_per_axis);
			return params;
		}

		cellmate_helpers::PointCloud make_points(const Config& config, const Params& params)
		{
			const std::size_t N_axis = params.N_clusters_per_axis;
			const std::size_t N_clusters = N_axis * N_axis * N_axis;
			const double cell_side = 2 * double(params.half_side) / N_axis;
			const double first_center = -double(params.half_side) + cell_side / 2;
			cellmate_helpers::PointCloud points;
			points.x.resize(config.N_points);
			points.y.resize(config.N_points);
			points.z.resize(config.N_points);
			std::mt19937 random(config.rng_seed);
			// using 'mt19937' with multiply-shift instead of distributions for identical clouds across standard libraries
			// (unbiased for power-of-two cluster counts)
			const auto cluster = [&random, N_clusters]() { return std::size_t((std::uint64_t(random()) * N_clusters) >> 32); };
			const auto sample = [&random]() { return 2 * double(random() >> 8) / 16777216.0 - 1; };
			const float last = std::nextafter(params.half_side, -params.half_side);
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				const std::size_t idx = cluster();
				const std::size_t ix = idx % N_axis;
				const std::size_t iy = (idx / N_axis) % N_axis;
				const std::size_t iz = idx / (N_axis * N_axis);
				double x, y, z;
				// rejection sampling gives uniform volume density inside the sphere
				do
				{
					x = sample();
					y = sample();
					z = sample();
				} while (x * x + y * y + z * z > 1);
				points.x[i] = (std::min)(last, float(first_center + ix * cell_side + params.cluster_radius * x));
				points.y[i] = (std::min)(last, float(first_center + iy * cell_side + params.cluster_radius * y));
				points.z[i] = (std::min)(last, float(first_center + iz * cell_side + params.cluster_radius * z));
			}
			return points;
		}
	}

	const Scenario& clustered_volume()
	{
		static const Scenario scenario{ "clustered_volume", [params = Params{}](const Config& config)
		{
			const auto derived = derive_params(config, params);
			return Workload{ make_points(config, derived), {},
				{ derived.half_side, derived.half_side, derived.half_side } };
		} };
		return scenario;
	}
}
