#pragma once

#include "../bench_matrix.hpp"

#include <vector>

namespace cellmate_bench
{
	// every library has its own parallelization
	struct ComparisonModeValues
	{
		std::vector<int> N_threads;
		std::vector<QueryPlacement> query_placement;
	};

	struct ComparisonMatrix
	{
		Config reference;
		ScenarioValues scenarios;
		ComparisonModeValues modes;
		OperationValues operations;
		SweepValues sweeps;
	};
}
