#include <cellmate/neighbor_search.hpp>
#include <benchmark/benchmark.h>

#include "../helpers/point_cloud.hpp"

#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cellmate_bench
{
	void configure(benchmark::internal::Benchmark* bench)
	{
		bench
			->Arg(1000)
			->Arg(10000)
			->Arg(100000)
			->UseRealTime()
			->Unit(benchmark::kMicrosecond)
			->MinTime(1.0)
			->Repetitions(3);
	}

#ifdef _OPENMP
	void prepare_parallel(const benchmark::State&)
	{
		// TODO: pick thread count differently?
		omp_set_dynamic(0);
		omp_set_num_threads(8);
	}
#endif

	void rebuild(benchmark::State& state, cellmate::Exec exec)
	{
		// prepare input data and search object
		const std::size_t N_points = static_cast<std::size_t>(state.range(0));
		const cellmate_helpers::PointCloud points = cellmate_helpers::make_uniform_cloud(N_points, 67, -5.0f, 5.0f);
		cellmate::NeighborSearch search(1.0f);
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
	}

	// run sequential and parallel rebuild() benchmarks
	BENCHMARK_CAPTURE(rebuild, seq, cellmate::Exec::seq)->Apply(configure);
#ifdef _OPENMP
	BENCHMARK_CAPTURE(rebuild, par, cellmate::Exec::par)
		->Apply(configure)->Setup(prepare_parallel)->MeasureProcessCPUTime();
#endif

	void query(benchmark::State& state)
	{
		// prepare input data and search object
		const std::size_t N_points = static_cast<std::size_t>(state.range(0));
		const cellmate_helpers::PointCloud points = cellmate_helpers::make_uniform_cloud(N_points, 67, -5.0f, 5.0f);
		cellmate::NeighborSearch search(1.0f);
		search.rebuild(
			points.x.data(),
			points.y.data(),
			points.z.data(),
			points.x.size(),
			cellmate::Exec::seq
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

	// run sequential query() benchmarks
	BENCHMARK(query)->Apply(configure);

	void query_batch(benchmark::State& state, cellmate::Exec exec)
	{
		// prepare input data and search object
		const std::size_t N_points = static_cast<std::size_t>(state.range(0));
		const cellmate_helpers::PointCloud points = cellmate_helpers::make_uniform_cloud(N_points, 67, -5.0f, 5.0f);
		cellmate::NeighborSearch search(1.0f);
		search.rebuild(
			points.x.data(),
			points.y.data(),
			points.z.data(),
			points.x.size(),
			cellmate::Exec::seq
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
	}

	// run sequential and parallel query_batch() benchmarks
	BENCHMARK_CAPTURE(query_batch, seq, cellmate::Exec::seq)->Apply(configure);
#ifdef _OPENMP
	BENCHMARK_CAPTURE(query_batch, par, cellmate::Exec::par)
		->Apply(configure)->Setup(prepare_parallel)->MeasureProcessCPUTime();
#endif
}

BENCHMARK_MAIN();
