#include "../bench_config.hpp"

#include <algorithm>
#include <cmath>

namespace cellmate_bench
{
	namespace
	{
		struct Params
		{
			float half_side = 0.0f;
		};

		Params derive_params(const Config& config, Params params)
		{
			params.half_side = float(search_radius * std::sqrt(
				double(config.N_points) * pi / config.N_nominal_avg_nbrs)) / 2;
			return params;
		}

		cellmate_helpers::PointCloud make_points(const Config& config, const Params& params)
		{
			auto points = cellmate_helpers::make_uniform_cloud(
				config.N_points, config.rng_seed, -params.half_side, params.half_side);
			std::fill(points.z.begin(), points.z.end(), 0.0f);
			return points;
		}
	}

	const Scenario& planar_surface()
	{
		static const Scenario scenario{ "planar_surface", Config{}, [params = Params{}](const Config& config)
		{
			const auto derived = derive_params(config, params);
			return Workload{ make_points(config, derived), {},
				{ derived.half_side, derived.half_side, 0.0f } };
		} };
		return scenario;
	}
}
