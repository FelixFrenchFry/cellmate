#pragma once

#include "../comparison_matrix.hpp"

namespace cellmate_bench
{
	inline ComparisonMatrix comparison_matrix()
	{
		ComparisonMatrix matrix;
		matrix.reference.N_points = 1'000'000;
		matrix.reference.N_queries = 1'000'000;
		matrix.reference.N_target_avg_nbrs = 32.0;
		matrix.reference.N_max_threads = 8;
		matrix.reference.rng_seed = 67;

		// combined sweeps through the parameter space
		matrix.scenarios.values = {
			&uniform_volume(),
			&clustered_volume(),
			&planar_surface(),
			&spherical_surface()
		};
		matrix.modes.N_threads = {
			1,
			8
		};
		matrix.modes.query_placement = {
			QueryPlacement::points
		};
		matrix.operations.values = {
			{ "rebuild",             RebuildMode::once, QueryMode::none },
			{ "query",               RebuildMode::none, QueryMode::single },
			{ "query_batch",         RebuildMode::none, QueryMode::batch },
			{ "rebuild_query_batch", RebuildMode::once, QueryMode::batch }
		};

		// individual sweeps through the parameter space
		matrix.sweeps.N_points = {
			1000,
			10000,
			100000,
			1000000, // 1 mio
		};
		matrix.sweeps.N_queries = {
			10,
			100,
			1000,
			10000,
			100000,
			1000000, // 1 mio
		};
		matrix.sweeps.N_target_avg_nbrs = {
			0.5,
			2.0,
			8.0,
			32.0,
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
