#include "../bench_config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cellmate_bench
{
	namespace
	{
		struct Params
		{
			std::size_t N_slabs = 10;
			float half_side = 0.0f;
			double total_weight = 0.0;
		};

		Params derive_params(const Config& config, Params params)
		{
			if (params.N_slabs == 0)
			{
				throw std::invalid_argument("slab count must be positive");
			}
			
			params.half_side = float(search_radius * std::cbrt(
				double(config.N_points) * (4.0 / 3) * pi / config.N_target_avg_nbrs)) / 2;
			params.total_weight = double(params.N_slabs) * (double(params.N_slabs) + 1) / 2;
			return params;
		}

		cellmate_helpers::PointCloud make_points(const Config& config, const Params& params)
		{
			if (params.N_slabs == 1)
			{
				return uniform_volume().make_workload(config).points;
			}

			auto points = cellmate_helpers::make_uniform_cloud(config.N_points, config.rng_seed, 0.0f, 1.0f);
			const double extent = double(params.half_side) * 2;
			const float last = std::nextafter(params.half_side, -params.half_side);

			for (std::size_t i = 0; i < config.N_points; i++)
			{
				// select a slab with weights [1, 2, ..., N_slabs] and sample uniformly inside it
				double ux = points.x[i] * params.total_weight;
				double weight = 1;
				std::size_t slab = 0;
				while (slab + 1 < params.N_slabs && ux >= weight)
				{
					ux -= weight;
					weight++;
					slab++;
				}
				ux = (slab + ux / weight) / params.N_slabs;
				points.x[i] = (std::min)(last, float(-double(params.half_side) + extent * ux));
				points.y[i] = (std::min)(last, float(-double(params.half_side) + extent * points.y[i]));
				points.z[i] = (std::min)(last, float(-double(params.half_side) + extent * points.z[i]));
			}
			return points;
		}
	}

	const Scenario& gradient_volume()
	{
		static const Scenario scenario{ "gradient_volume", [params = Params{}](const Config& config)
		{
			const auto derived = derive_params(config, params);
			return Workload{ make_points(config, derived), {},
				{ derived.half_side, derived.half_side, derived.half_side } };
		} };
		return scenario;
	}
}
