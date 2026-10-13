#include <cellmate/neighbor_search.hpp>
#include <benchmark/benchmark.h>

#include "bench_config.hpp"
#include "../helpers/brute_force.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cellmate_bench
{
	namespace
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
			std::vector<const Scenario*> values{
				&uniform_volume(),
				&gradient_volume(),
				&planar_surface()
			};
		};

		struct ModeValues
		{
			std::vector<cellmate::ComputeMode> compute_mode{
				cellmate::ComputeMode::seq,
				cellmate::ComputeMode::par
			};
			std::vector<cellmate::PointAccess> point_access{
				cellmate::PointAccess::cpy
			};
			std::vector<cellmate::QueryOrder> query_order{
				cellmate::QueryOrder::morton
			};
			std::vector<QueryPlacement> query_placement{
				QueryPlacement::points
			};
		};

		struct OperationValues
		{
			std::vector<Operation> values{
				{ "rebuild",             RebuildMode::once, QueryMode::none },
				{ "query",               RebuildMode::none, QueryMode::single },
				{ "query_batch",         RebuildMode::none, QueryMode::batch },
				{ "rebuild_query",       RebuildMode::once, QueryMode::single },
				{ "rebuild_query_batch", RebuildMode::once, QueryMode::batch }
			};
		};

		struct SweepValues
		{
			std::vector<std::size_t> N_points{
				1,
				10,
				100,
				1000,
				10000,
				100000,
				1000000, // 1 mio
				10000000
			};
			std::vector<std::size_t> N_queries{
				1,
				10,
				100,
				1000,
				10000,
				100000,
				1000000, // 1 mio
				10000000
			};
			std::vector<double> N_nominal_avg_nbrs{
				0.1,
				0.5,
				1.0,
				2.0,
				4.0,
				8.0,
				16.0,
				32.0,
				64.0,
				128.0,
				256.0
			};
			std::vector<int> N_max_threads{
				1,
				2,
				4,
				8
			};
		};

#ifdef _OPENMP
		class OpenMPSettings
		{
		public:

			explicit OpenMPSettings(int max_threads)
				: previous_threads_(omp_get_max_threads()), previous_dynamic_(omp_get_dynamic())
			{
				omp_set_dynamic(0);
				omp_set_num_threads(max_threads);
			}

			~OpenMPSettings()
			{
				omp_set_num_threads(previous_threads_);
				omp_set_dynamic(previous_dynamic_);
			}

		private:

			int previous_threads_, previous_dynamic_;
		};
#endif

		void run(benchmark::State& state, const Scenario& scenario, const Config& config,
			const Operation& operation, bool& failed)
		{
			try
			{
#ifndef _OPENMP
				if (config.compute_mode == cellmate::ComputeMode::par)
				{
					state.SkipWithError("parallel benchmarks require CELLMATE_ENABLE_OPENMP=ON");
					failed = true;
					return;
				}
#endif
				const auto workload = prepare_workload(scenario, config);
				const auto& points = workload.points;
				const auto& queries = workload.queries;
#ifdef _OPENMP
				const OpenMPSettings settings(config.N_max_threads);
#endif
				cellmate::NeighborSearch search(search_radius, config.point_access);
				std::vector<std::size_t> neighbors, offsets;
				std::size_t N_neighbors = 0;
				const auto rebuild = [&]()
				{
					search.rebuild(points.x.data(), points.y.data(), points.z.data(),
						config.N_points, config.compute_mode);
				};
				const auto run_queries = [&]()
				{
					N_neighbors = 0;
					switch (operation.query_mode)
					{
					case QueryMode::none:
						break;
					case QueryMode::single:
						for (std::size_t i = 0; i < config.N_queries; i++)
						{
							search.query({ queries.x[i], queries.y[i], queries.z[i] }, neighbors);
							N_neighbors += neighbors.size();
							benchmark::DoNotOptimize(neighbors.data());
							benchmark::ClobberMemory();
						}
						break;
					case QueryMode::batch:
						search.query_batch(queries.x.data(), queries.y.data(), queries.z.data(),
							config.N_queries, neighbors, offsets, config.compute_mode, config.query_order);
						N_neighbors = neighbors.size();
						break;
					}
				};
				const auto validate = [&]()
				{
					if (operation.query_mode == QueryMode::batch &&
						(offsets.size() != config.N_queries + 1 || offsets.front() != 0 ||
							offsets.back() != neighbors.size() || !std::is_sorted(offsets.begin(), offsets.end())))
					{
						throw std::runtime_error("batch offsets do not describe the returned neighbors");
					}
					// compare complete memberships at up to eight centers spread through the cloud
					const std::size_t N_samples = (std::min)(config.N_queries, std::size_t(8));
					std::vector<std::size_t> actual, expected;
					for (std::size_t i = 0; i < N_samples; i++)
					{
						const std::size_t idx = config.N_queries <= 8 ? i : i * (config.N_queries - 1) / 7;
						const std::array<float, 3> query_pos{ queries.x[idx], queries.y[idx], queries.z[idx] };
						if (operation.query_mode != QueryMode::batch)
						{
							search.query(query_pos, actual);
						}
						else
						{
							actual.assign(neighbors.begin() + offsets[idx], neighbors.begin() + offsets[idx + 1]);
						}
						cellmate_helpers::brute_force(points.x.data(), points.y.data(), points.z.data(),
							config.N_points, search_radius, query_pos, expected);
						std::sort(actual.begin(), actual.end());
						if (actual != expected)
						{
							throw std::runtime_error("neighbor IDs do not match brute force");
						}
					}
				};

				// initialize reusable storage and check correctness outside timing
				rebuild();
				run_queries();
				validate();

				for (auto _ : state)
				{
					if (operation.rebuild_mode == RebuildMode::once) { rebuild(); }
					run_queries();
					benchmark::DoNotOptimize(search);
					benchmark::DoNotOptimize(neighbors.data());
					benchmark::DoNotOptimize(offsets.data());
					benchmark::ClobberMemory();
				}

				validate();
				state.counters["N_points"] = double(config.N_points);
				state.counters["N_queries"] = double(config.N_queries);
				state.counters["N_nominal_avg_nbrs"] = config.N_nominal_avg_nbrs;
				state.counters["radius"] = search_radius;
				state.counters["N_max_threads"] = config.N_max_threads;
				state.counters["rng_seed"] = config.rng_seed;
				state.counters["parallel_compute"] = config.compute_mode == cellmate::ComputeMode::par;
				state.counters["copied_points"] = config.point_access == cellmate::PointAccess::cpy;
				state.counters["morton_queries"] = config.query_order == cellmate::QueryOrder::morton;
				state.counters["query_placement"] = int(config.query_placement);
				if (operation.query_mode != QueryMode::none)
				{
					state.counters["neighbors"] = double(N_neighbors);
					state.counters["mean_neighbors"] = double(N_neighbors) / config.N_queries;
					state.counters["N_avg_nbrs"] = double(N_neighbors) / config.N_queries -
						(config.query_placement == QueryPlacement::points ? 1 : 0);
					state.counters["queries_per_second"] = benchmark::Counter(
						double(config.N_queries), benchmark::Counter::kIsIterationInvariantRate);
				}
			}
			catch (const std::exception& error)
			{
				state.SkipWithError(error.what());
				failed = true;
			}
		}

		template<typename T>
		void register_sweep(const Scenario* scenario, const Config& reference, const Operation& operation,
			const char* experiment, const char* parameter_name, T Config::* member,
			const std::vector<T>& values, bool& failed)
		{
			for (T value : values)
			{
				Config config = reference;
				config.*member = value;
				std::ostringstream name;
				name << scenario->name << '/' << experiment << '/' << operation.name << '/' << parameter_name << ':' << value
					<< "/compute_mode:" << (config.compute_mode == cellmate::ComputeMode::par ? "par" : "seq")
					<< "/point_access:" << (config.point_access == cellmate::PointAccess::cpy ? "cpy" : "ref")
					<< "/query_order:" << (config.query_order == cellmate::QueryOrder::morton ? "morton" : "input")
					<< "/query_placement:" << (config.query_placement == QueryPlacement::points ? "points" : "domain");
				auto* registration = benchmark::RegisterBenchmark(name.str().c_str(),
					[scenario, operation, config, &failed](benchmark::State& state)
					{
						run(state, *scenario, config, operation, failed);
					});
				registration->UseRealTime()->Unit(benchmark::kMillisecond);
			}
		}

		void register_experiments(const Scenario* scenario, Config reference, const Operation& operation,
			const SweepValues& sweeps, bool& failed)
		{
			if (reference.compute_mode == cellmate::ComputeMode::seq) { reference.N_max_threads = 1; }

			register_sweep(scenario, reference, operation, "reference", "N",
				&Config::N_points, { reference.N_points }, failed);
			register_sweep(scenario, reference, operation, "points", "N",
				&Config::N_points, sweeps.N_points, failed);
			register_sweep(scenario, reference, operation, "queries", "N_queries",
				&Config::N_queries, sweeps.N_queries, failed);
			register_sweep(scenario, reference, operation, "neighbors", "N_nominal_avg_nbrs",
				&Config::N_nominal_avg_nbrs, sweeps.N_nominal_avg_nbrs, failed);
			if (reference.compute_mode == cellmate::ComputeMode::par)
			{
				register_sweep(scenario, reference, operation, "threads", "N_max_threads",
					&Config::N_max_threads, sweeps.N_max_threads, failed);
			}
		}
	}

	void register_benchmarks(bool& failed)
	{
		const SweepValues sweeps;
		const ModeValues modes;
		const ScenarioValues scenarios;
		const OperationValues operations;

		// iterate over entire parameter matrix
		for (const Scenario* scenario : scenarios.values)
		for (auto compute_mode : modes.compute_mode)
		for (auto point_access : modes.point_access)
		for (auto query_order : modes.query_order)
		for (auto query_placement : modes.query_placement)
		{
			Config reference;
			reference.compute_mode = compute_mode;
			reference.point_access = point_access;
			reference.query_order = query_order;
			reference.query_placement = query_placement;

			for (const Operation& operation : operations.values)
			{
				register_experiments(scenario, reference, operation, sweeps, failed);
			}
		}
	}
}

int main(int argc, char** argv)
{
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv)) { return 1; }
	bool failed = false;
	cellmate_bench::register_benchmarks(failed);
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
	return failed ? 1 : 0;
}
