#include <cellmate/neighbor_search.hpp>
#include <benchmark/benchmark.h>

#include "bench_config.hpp"

#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cellmate_bench
{
#ifdef _OPENMP
	const int max_threads = omp_get_max_threads();

	void prepare_parallel(const benchmark::State&)
	{
		// restore the configured thread limit after serial benchmarks
		omp_set_dynamic(0);
		omp_set_num_threads(max_threads);
	}
#endif

	void rebuild(benchmark::State& state, cellmate::ComputeMode exec)
	{
		// prepare input data and search object
		const cellmate_helpers::PointCloud points = make_points(state);
		cellmate::NeighborSearch search(search_radius);
		search.rebuild(
			points.x.data(),
			points.y.data(), 
			points.z.data(), 
			points.x.size(), 
			exec
		);

		// run benchmark loop
		for (auto _ : state)
		{
			search.rebuild(
				points.x.data(),
				points.y.data(), 
				points.z.data(), 
				points.x.size(), 
				exec
			);

			benchmark::DoNotOptimize(search);
			benchmark::ClobberMemory();
		}

#ifdef _OPENMP
		if (exec == cellmate::ComputeMode::par) { state.counters["max_threads"] = max_threads; }
#endif
	}

	// run sequential and parallel 'rebuild()' benchmarks
	BENCHMARK_CAPTURE(rebuild, seq, cellmate::ComputeMode::seq)->Apply(configure);
#ifdef _OPENMP
	BENCHMARK_CAPTURE(rebuild, par, cellmate::ComputeMode::par)
		->Apply(configure)->Setup(prepare_parallel)->MeasureProcessCPUTime();
#endif

	void query(benchmark::State& state)
	{
		// prepare input data and search object
		const cellmate_helpers::PointCloud points = make_points(state);
		cellmate::NeighborSearch search(search_radius);
		search.rebuild(
			points.x.data(),
			points.y.data(),
			points.z.data(),
			points.x.size(),
			cellmate::ComputeMode::seq
		);

		// prepare reusable output storage and query cache
		std::vector<std::size_t> neighbors;
		for (std::size_t i = 0; i < points.x.size(); i++)
		{
			search.query({ points.x[i], points.y[i], points.z[i] }, neighbors);
		}

		// run benchmark loop
		for (auto _ : state)
		{
			for (std::size_t i = 0; i < points.x.size(); i++)
			{
				search.query({ points.x[i], points.y[i], points.z[i] }, neighbors);
				benchmark::DoNotOptimize(neighbors);
				benchmark::ClobberMemory();
			}
		}
	}

	// run sequential 'query()' benchmarks
	BENCHMARK(query)->Name("query/cellmate/seq")->Apply(configure);

	void query_batch(benchmark::State& state, cellmate::ComputeMode exec)
	{
		// prepare input data and search object
		const cellmate_helpers::PointCloud points = make_points(state);
		cellmate::NeighborSearch search(search_radius);
		search.rebuild(
			points.x.data(),
			points.y.data(),
			points.z.data(),
			points.x.size(),
			cellmate::ComputeMode::seq
		);

		// prepare reusable output storage and query cache
		std::vector<std::size_t> neighbors, offsets;
		search.query_batch(
			points.x.data(),
			points.y.data(),
			points.z.data(),
			points.x.size(),
			neighbors,
			offsets,
			exec
		);

		// run benchmark loop
		for (auto _ : state)
		{
			search.query_batch(
				points.x.data(),
				points.y.data(),
				points.z.data(),
				points.x.size(),
				neighbors,
				offsets,
				exec
			);

			benchmark::DoNotOptimize(neighbors);
			benchmark::DoNotOptimize(offsets);
			benchmark::ClobberMemory();
		}

#ifdef _OPENMP
		if (exec == cellmate::ComputeMode::par) { state.counters["max_threads"] = max_threads; }
#endif
	}

	// run sequential and parallel 'query_batch()' benchmarks
	BENCHMARK_CAPTURE(query_batch, seq, cellmate::ComputeMode::seq)->Apply(configure);
#ifdef _OPENMP
	BENCHMARK_CAPTURE(query_batch, par, cellmate::ComputeMode::par)
		->Apply(configure)->Setup(prepare_parallel)->MeasureProcessCPUTime();
#endif
}
