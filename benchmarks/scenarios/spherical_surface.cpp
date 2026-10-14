#include "../bench_config.hpp"

#include <cmath>

namespace cellmate_bench
{
	namespace
	{
		struct Params
		{
			float sphere_radius = 0.0f;
		};

		Params derive_params(const Config& config, Params params)
		{
			params.sphere_radius = float(search_radius * std::sqrt(
				double(config.N_points) / (4 * config.N_target_avg_nbrs)));
			return params;
		}

		cellmate_helpers::PointCloud make_points(const Config& config, const Params& params)
		{
			auto points = cellmate_helpers::make_uniform_cloud(config.N_points, config.rng_seed, 0.0f, 1.0f);
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				// uniform height and azimuth give uniform surface area without polar clustering
				const double z = 2 * double(points.z[i]) - 1;
				const double angle = 2 * pi * points.x[i];
				const double xy = params.sphere_radius * std::sqrt(1 - z * z);
				points.x[i] = float(xy * std::cos(angle));
				points.y[i] = float(xy * std::sin(angle));
				points.z[i] = float(params.sphere_radius * z);
			}
			return points;
		}
	}

	const Scenario& spherical_surface()
	{
		static const Scenario scenario{ "spherical_surface", [params = Params{}](const Config& config)
		{
			const auto derived = derive_params(config, params);
			return Workload{ make_points(config, derived), {},
				{ derived.sphere_radius, derived.sphere_radius, derived.sphere_radius } };
		} };
		return scenario;
	}
}
