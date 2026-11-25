#pragma once

#include "../comparison_matrix.hpp"

namespace cellmate_bench
{
	inline ComparisonMatrix comp_threads_matrix()
	{
		ComparisonMatrix matrix;
		matrix.reference.N_points = 500'000;
		matrix.reference.N_queries = 500'000;
		matrix.reference.N_target_avg_nbrs = 32.0;
		matrix.reference.N_max_threads = 8;
		matrix.reference.rng_seed = 67;

		// combined sweeps through the parameter space
		matrix.scenarios.values = {
			&uniform_volume()
		};
		matrix.modes.N_threads = {
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
			// nothing
		};
		matrix.sweeps.N_queries = {
			// nothing
		};
		matrix.sweeps.N_target_avg_nbrs = {
			// nothing
		};
		matrix.sweeps.N_max_threads = {
			1,
			2,
			3,
			4,
			5,
			6,
			7,
			8,
		};
		return matrix;
	}
}
