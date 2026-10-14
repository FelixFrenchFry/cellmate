#pragma once

#include "../bench_matrix.hpp"

namespace cellmate_bench
{
	inline BenchmarkMatrix single_matrix()
	{
		BenchmarkMatrix matrix;
		matrix.reference.N_points = 1'000'000;
		matrix.reference.N_queries = 1'000'000;
		matrix.reference.N_target_avg_nbrs = 32.0;
		matrix.reference.N_max_threads = 8;
		matrix.reference.rng_seed = 67;

		// combined sweeps through the parameter space
		matrix.scenarios.values = {
			&uniform_volume()
		};
		matrix.modes.compute_mode = {
			cellmate::ComputeMode::par
		};
		matrix.modes.point_access = {
			cellmate::PointAccess::cpy
		};
		matrix.modes.query_order = {
			cellmate::QueryOrder::morton
		};
		matrix.modes.query_placement = {
			QueryPlacement::points
		};
		matrix.operations.values = {
			{ "rebuild",             RebuildMode::once, QueryMode::none },
			{ "query",               RebuildMode::none, QueryMode::single },
			{ "query_batch",         RebuildMode::none, QueryMode::batch }
		};

		// individual sweeps through the parameter space
		matrix.sweeps.N_points = {
			100,
			10000,
			1000000, // 1 mio
		};
		matrix.sweeps.N_queries = {
			100,
			10000,
			1000000, // 1 mio
		};
		matrix.sweeps.N_target_avg_nbrs = {
			0.5,
			8.0,
			128.0
		};
		matrix.sweeps.N_max_threads = {
			// nothing
		};
		return matrix;
	}
}
