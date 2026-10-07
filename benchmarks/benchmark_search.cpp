#include <cellmate/neighbor_search.hpp>
#include <benchmark/benchmark.h>

#include "../helpers/point_cloud.hpp"

#include <cstddef>
#include <vector>

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

	void rebuild(benchmark::State& state)
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

		// run benchmark loop
		for (auto _ : state)
		{
			search.rebuild(
				points.x.data(),
				points.y.data(), 
				points.z.data(), 
				points.x.size(), 
				cellmate::Exec::seq
			);

			benchmark::DoNotOptimize(search);
			benchmark::ClobberMemory();
		}
	}

	BENCHMARK(rebuild)->Apply(configure);

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

	BENCHMARK(query)->Apply(configure);

	void query_batch(benchmark::State& state)
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
			cellmate::Exec::seq
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
				cellmate::Exec::seq
			);

			benchmark::DoNotOptimize(neighbors);
			benchmark::DoNotOptimize(offsets);
			benchmark::ClobberMemory();
		}
	}

	BENCHMARK(query_batch)->Apply(configure);
}

BENCHMARK_MAIN();
