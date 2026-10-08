#pragma once

#include <benchmark/benchmark.h>

#include "../helpers/point_cloud.hpp"

#include <cstddef>

namespace cellmate_bench
{
	constexpr float search_radius = 1.0f;

#ifdef _OPENMP
	void prepare_parallel(const benchmark::State&);
#endif

	inline cellmate_helpers::PointCloud make_points(const benchmark::State& state)
	{
		const std::size_t N_points = static_cast<std::size_t>(state.range(0));
		return cellmate_helpers::make_uniform_cloud(N_points, 67, -5.0f, 5.0f);
	}

	inline void configure(benchmark::internal::Benchmark* bench)
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
}
