#pragma once

#include "bench_config.hpp"

#include <cstddef>
#include <vector>

namespace cellmate_bench
{
	enum class RebuildMode
	{
		none,
		once
	};

	enum class QueryMode
	{
		none,
		single,
		batch
	};

	struct Operation
	{
		const char* name;
		RebuildMode rebuild_mode;
		QueryMode query_mode;
	};

	struct ScenarioValues
	{
		std::vector<const Scenario*> values;
	};

	struct ModeValues
	{
		std::vector<cellmate::ComputeMode> compute_mode;
		std::vector<cellmate::PointAccess> point_access;
		std::vector<cellmate::QueryOrder> query_order;
		std::vector<QueryPlacement> query_placement;
	};

	struct OperationValues
	{
		std::vector<Operation> values;
	};

	struct SweepValues
	{
		std::vector<std::size_t> N_points;
		std::vector<std::size_t> N_queries;
		std::vector<double> N_nominal_avg_nbrs;
		std::vector<int> N_max_threads;
	};

	struct BenchmarkMatrix
	{
		Config reference;
		ScenarioValues scenarios;
		ModeValues modes;
		OperationValues operations;
		SweepValues sweeps;
	};
}
