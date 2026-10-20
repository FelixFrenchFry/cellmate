#pragma once

#include "../comparison_matrix.hpp"
#include "../../profiles/quick.hpp"

namespace cellmate_bench
{
	// derived from the maintained 'quick' profile, with thread counts instead of compute modes
	inline ComparisonMatrix quick_comparison_matrix()
	{
		const BenchmarkMatrix maintained = quick_matrix();
		ComparisonMatrix matrix;
		matrix.reference = maintained.reference;
		matrix.scenarios = maintained.scenarios;
		matrix.operations = maintained.operations;
		matrix.sweeps = maintained.sweeps;

		// sequential and parallel runs, using the reference thread count
		matrix.modes.N_threads = {
			1,
			maintained.reference.N_max_threads
		};
		matrix.modes.query_placement = maintained.modes.query_placement;
		return matrix;
	}
}
