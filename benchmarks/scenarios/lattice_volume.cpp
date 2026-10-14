#include "../bench_config.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace cellmate_bench
{
	namespace
	{
		struct Params
		{
			std::size_t N_axis = 0;
			float half_side = 0.0f;
		};

		Params derive_params(const Config& config, Params params)
		{
			params.N_axis = std::size_t(std::cbrt(double(config.N_points)));
			while (params.N_axis * params.N_axis * params.N_axis < config.N_points) { params.N_axis++; }
			params.half_side = float(params.N_axis) / 2;
			return params;
		}

		cellmate_helpers::PointCloud make_points(const Config& config, const Params& params)
		{
			const std::size_t N_sites = params.N_axis * params.N_axis * params.N_axis;
			cellmate_helpers::PointCloud points;
			points.x.resize(N_sites);
			points.y.resize(N_sites);
			points.z.resize(N_sites);
			for (std::size_t i = 0; i < N_sites; i++)
			{
				// unit spacing keeps the initial lattice distances exact before calibration
				points.x[i] = float(i % params.N_axis) + 0.5f - params.half_side;
				points.y[i] = float((i / params.N_axis) % params.N_axis) + 0.5f - params.half_side;
				points.z[i] = float(i / (params.N_axis * params.N_axis)) + 0.5f - params.half_side;
			}
			std::mt19937 random(config.rng_seed);
			// shuffle and select distinct sites without giving the lattice sorted input
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				const std::size_t idx = i + cellmate_helpers::random_index(random, N_sites - i);
				std::swap(points.x[i], points.x[idx]);
				std::swap(points.y[i], points.y[idx]);
				std::swap(points.z[i], points.z[idx]);
			}
			points.x.resize(config.N_points);
			points.y.resize(config.N_points);
			points.z.resize(config.N_points);
			return points;
		}
	}

	const Scenario& lattice_volume()
	{
		static const Scenario scenario{ "lattice_volume", [params = Params{}](const Config& config)
		{
			const auto derived = derive_params(config, params);
			return Workload{ make_points(config, derived), {},
				{ derived.half_side, derived.half_side, derived.half_side } };
		} };
		return scenario;
	}
}
