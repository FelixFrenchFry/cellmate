#include "../bench_config.hpp"

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
			params.half_side = float(search_radius * std::cbrt(
				double(config.N_points) * (4.0 / 3) * pi / config.N_target_avg_nbrs)) / 2;
			return params;
		}

		cellmate_helpers::PointCloud make_points(const Config& config, const Params& params)
		{
			return cellmate_helpers::make_uniform_cloud(
				config.N_points, config.rng_seed, -params.half_side, params.half_side);
		}
	}

	const Scenario& uniform_volume()
	{
		static const Scenario scenario{ "uniform_volume", [params = Params{}](const Config& config)
		{
			const auto derived = derive_params(config, params);
			return Workload{ make_points(config, derived), {},
				{ derived.half_side, derived.half_side, derived.half_side } };
		} };
		return scenario;
	}
}
