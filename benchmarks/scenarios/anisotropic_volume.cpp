#include "../bench_config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace cellmate_bench
{
	namespace
	{
		struct Params
		{
			std::array<double, 3> aspect_ratio{ 64.0, 1.0, 1.0 };
			std::array<float, 3> half_extent{};
		};

		Params derive_params(const Config& config, Params params)
		{
			double volume_ratio = 1;
			for (double ratio : params.aspect_ratio)
			{
				if (!std::isfinite(ratio) || ratio <= 0)
				{
					throw std::invalid_argument("aspect ratios must be finite and positive");
				}
				volume_ratio *= ratio;
			}
			// preserve the uniform cube's initial volume while changing its proportions
			const double half_side = search_radius * std::cbrt(
				double(config.N_points) * (4.0 / 3) * pi / (config.N_target_avg_nbrs * volume_ratio)) / 2;
			for (std::size_t axis = 0; axis < params.half_extent.size(); axis++)
			{
				params.half_extent[axis] = float(half_side * params.aspect_ratio[axis]);
			}
			return params;
		}

		cellmate_helpers::PointCloud make_points(const Config& config, const Params& params)
		{
			auto points = cellmate_helpers::make_uniform_cloud(config.N_points, config.rng_seed, -1.0f, 1.0f);
			std::size_t axis = 0;
			for (auto* coordinates : { &points.x, &points.y, &points.z })
			{
				const float half_side = params.half_extent[axis++];
				const float last = std::nextafter(half_side, -half_side);
				for (float& value : *coordinates)
				{
					value = (std::min)(last, float(double(value) * half_side));
				}
			}
			return points;
		}
	}

	const Scenario& anisotropic_volume()
	{
		static const Scenario scenario{ "anisotropic_volume", [params = Params{}](const Config& config)
		{
			const auto derived = derive_params(config, params);
			return Workload{ make_points(config, derived), {}, derived.half_extent };
		} };
		return scenario;
	}
}
