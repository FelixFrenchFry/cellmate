#pragma once

#include "../bench_matrix.hpp"

namespace cellmate_bench
{
	inline BenchmarkMatrix balanced_matrix()
	{
		BenchmarkMatrix matrix;
		matrix.reference.N_points = 1'000'000;
		matrix.reference.N_queries = 1'000'000;
		matrix.reference.N_nominal_avg_nbrs = 32.0;
		matrix.reference.N_max_threads = 8;
		matrix.reference.rng_seed = 67;

		// combined sweeps through the parameter space
		matrix.scenarios.values = {
			&uniform_volume(),
			&gradient_volume(),
			&clustered_volume(),
			&planar_surface()
		};
		matrix.modes.compute_mode = {
			cellmate::ComputeMode::seq,
			cellmate::ComputeMode::par
		};
		matrix.modes.point_access = {
			cellmate::PointAccess::cpy,
			cellmate::PointAccess::ref
		};
		matrix.modes.query_order = {
			cellmate::QueryOrder::morton,
			cellmate::QueryOrder::input
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
			1,
			10,
			100,
			1000,
			10000,
			100000,
			1000000, // 1 mio
			10000000
		};
		matrix.sweeps.N_queries = {
			1,
			10,
			100,
			1000,
			10000,
			100000,
			1000000, // 1 mio
			10000000
		};
		matrix.sweeps.N_nominal_avg_nbrs = {
			0.1,
			0.5,
			1.0,
			2.0,
			4.0,
			8.0,
			16.0,
			32.0,
			64.0,
			128.0
		};
		matrix.sweeps.N_max_threads = {
			1,
			2,
			4,
			8
		};
		return matrix;
	}
}
