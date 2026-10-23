#include <cellmate/neighbor_search.hpp>
#include <benchmark/benchmark.h>

#include "comparison_revision.hpp"
#include "comparison_matrix.hpp"
#include "profiles/comparison.hpp"
#include "profiles/comp_points.hpp"
#include "profiles/comp_queries.hpp"
#include "profiles/comp_neighbors.hpp"
#include "profiles/comp_threads.hpp"
#include "profiles/comp_scenarios.hpp"
#include "adapters/cellmate.hpp"
#include "adapters/nanoflann.hpp"
#include "adapters/pico_tree.hpp"
#include "adapters/unibn_octree.hpp"
#include "adapters/compact_nsearch.hpp"
#include "adapters/ponca.hpp"
#include "adapters/cgal.hpp"
#include "adapters/boost_rtree.hpp"
#ifdef CELLMATE_COMPARISON_TREENSEARCH
#include "adapters/tree_nsearch.hpp"
#endif
#include "../../helpers/brute_force.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cellmate_bench
{
	namespace
	{
		// for benchmarks of the regular query() function, a small prefix of the queries
		// is enough to warm up the per-query storage and per-thread cache
		constexpr std::size_t N_warmup_queries = 10'000;

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

		template<typename Adapter>
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
				Adapter adapter(workload, config);
				typename Adapter::QueryOutput output;
				BatchResults results;
				LoopScratch<Adapter> scratch;
				std::size_t N_neighbors = 0;
				const auto run_queries = [&](std::size_t N_run)
				{
					N_neighbors = 0;
					switch (operation.query_mode)
					{
					case QueryMode::none:
						break;
					case QueryMode::single:
						if constexpr (Adapter::query_support != Support::none)
						{
							for (std::size_t i = 0; i < N_run; i++)
							{
								N_neighbors += adapter.query({ queries.x[i], queries.y[i], queries.z[i] }, output);
								benchmark::DoNotOptimize(output.data());
								benchmark::ClobberMemory();
							}
						}
						break;
					case QueryMode::batch:
						if constexpr (Adapter::batch_support == Support::native)
						{
							adapter.query_batch(N_run, results);
						}
						else
						{
							query_batch_loop(adapter, queries, N_run, config.compute_mode, config.N_max_threads, scratch, results);
						}
						break;
					}
				};
				// returns the total neighbor count of the last batch, outside timing since native formats may need a pass
				const auto count_batch_neighbors = [&]()
				{
					if constexpr (Adapter::batch_support == Support::native)
					{
						return adapter.batch_neighbor_count(results);
					}
					else
					{
						return batch_neighbor_count(results);
					}
				};
				const auto validate = [&]()
				{
					// compare complete memberships at up to eight centers spread through the cloud
					const std::size_t N_samples = (std::min)(config.N_queries, std::size_t(8));
					std::vector<std::uint32_t> actual;
					std::vector<std::uint32_t> expected;
					for (std::size_t i = 0; i < N_samples; i++)
					{
						const std::size_t idx = config.N_queries <= 8 ? i : i * (config.N_queries - 1) / 7;
						const std::array<float, 3> query_pos{ queries.x[idx], queries.y[idx], queries.z[idx] };
						if (operation.query_mode != QueryMode::batch)
						{
							if constexpr (Adapter::query_support != Support::none)
							{
								adapter.query(query_pos, output);
								actual.clear();
								adapter.append_indices(output, actual);
							}
						}
						else if constexpr (Adapter::batch_support == Support::native)
						{
							adapter.batch_neighbors(results, idx, actual);
						}
						else
						{
							batch_neighbors(results, idx, actual);
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

				// restores the input order outside timing for libraries that reorder their input on each rebuild,
				// so every timed rebuild starts from the same unsorted positions
				const auto prepare_rebuild = [&]()
				{
					if constexpr (Adapter::reorders_input)
					{
						adapter.prepare_rebuild();
					}
				};

				// initialize reusable storage and check correctness outside timing
				// (batches warm up with all queries to size their output and scratch buffers)
				prepare_rebuild();
				adapter.rebuild();
				run_queries(operation.query_mode == QueryMode::single
					? (std::min)(config.N_queries, N_warmup_queries) : config.N_queries);
				validate();

				for (auto _ : state)
				{
					if (operation.rebuild_mode == RebuildMode::once)
					{
						if constexpr (Adapter::reorders_input)
						{
							state.PauseTiming();
							prepare_rebuild();
							state.ResumeTiming();
						}
						adapter.rebuild();
					}
					run_queries(config.N_queries);
					benchmark::DoNotOptimize(adapter);
					benchmark::DoNotOptimize(output.data());
					benchmark::DoNotOptimize(results.neighbors.data());
					benchmark::DoNotOptimize(results.offsets.data());
					benchmark::ClobberMemory();
				}

				validate();
				if (operation.query_mode == QueryMode::batch) { N_neighbors = count_batch_neighbors(); }
				state.counters["N_points"] = double(config.N_points);
				state.counters["N_queries"] = double(config.N_queries);
				state.counters["N_target_avg_nbrs"] = config.N_target_avg_nbrs;
				state.counters["radius"] = search_radius;
				state.counters["N_max_threads"] = config.N_max_threads;
				state.counters["rng_seed"] = config.rng_seed;
				state.counters["parallel_compute"] = config.compute_mode == cellmate::ComputeMode::par;
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

		// WARNING: assumes rebuild-only operations validate with single queries, update when this changes
		template<typename Adapter>
		bool supports(const Operation& operation)
		{
			switch (operation.query_mode)
			{
			case QueryMode::none:
				return Adapter::rebuild_support != Support::none && Adapter::query_support != Support::none;
			case QueryMode::single:
				return Adapter::query_support != Support::none;
			case QueryMode::batch:
				// libraries rebuilding within every batch are only measured with rebuilds
				if (Adapter::batch_includes_rebuild)
				{
					return Adapter::batch_support != Support::none && operation.rebuild_mode == RebuildMode::once;
				}
				return Adapter::batch_support != Support::none &&
					(operation.rebuild_mode == RebuildMode::none || Adapter::rebuild_support != Support::none);
			}
			return false;
		}
		// WARNING: assumes single queries run on the calling thread, update when this changes
		template<typename Adapter>
		bool uses_threads(const Operation& operation)
		{
			return (operation.rebuild_mode == RebuildMode::once && Adapter::parallel_rebuild) ||
				operation.query_mode == QueryMode::batch;
		}
		// WARNING: assumes rebuilds never read query positions, update when this changes
		bool uses_queries(const Operation& operation)
		{
			return operation.query_mode != QueryMode::none;
		}

		template<typename Adapter, typename T>
		void register_sweep(const Scenario* scenario, const Config& reference, const Operation& operation,
			const char* experiment, const char* parameter_name, T Config::* member,
			const std::vector<T>& values, bool& failed, bool queries_follow_points = false)
		{
			for (T value : values)
			{
				Config config = reference;
				config.*member = value;
				if (queries_follow_points) { config.N_queries = config.N_points; }
				std::ostringstream name;
				name << scenario->name << '/' << experiment << '/' << operation.name << '/' << parameter_name << ':' << value
					<< "/library:" << Adapter::name
					<< "/threads:" << config.N_max_threads
					<< "/query_placement:" << (config.query_placement == QueryPlacement::points ? "points" : "domain");
				auto* registration = benchmark::RegisterBenchmark(name.str().c_str(),
					[scenario, operation, config, &failed](benchmark::State& state)
					{
						run<Adapter>(state, *scenario, config, operation, failed);
					});
				registration->UseRealTime()->Unit(benchmark::kMillisecond);
			}
		}

		template<typename Adapter>
		void register_experiments(const Scenario* scenario, Config reference, const Operation& operation,
			const SweepValues& sweeps, bool queries_follow_points, bool& failed)
		{
			// WARNING: assumes cellmate runs sequentially with one thread and in parallel otherwise, update when this changes
			reference.compute_mode = reference.N_max_threads > 1 ? cellmate::ComputeMode::par : cellmate::ComputeMode::seq;

			register_sweep<Adapter>(scenario, reference, operation, "reference", "N",
				&Config::N_points, { reference.N_points }, failed);
			register_sweep<Adapter>(scenario, reference, operation, "points", "N",
				&Config::N_points, sweeps.N_points, failed, queries_follow_points);
			if (uses_queries(operation))
			{
				register_sweep<Adapter>(scenario, reference, operation, "queries", "N_queries",
					&Config::N_queries, sweeps.N_queries, failed);
			}
			register_sweep<Adapter>(scenario, reference, operation, "neighbors", "N_target_avg_nbrs",
				&Config::N_target_avg_nbrs, sweeps.N_target_avg_nbrs, failed);
			if (reference.N_max_threads > 1 && uses_threads<Adapter>(operation))
			{
				register_sweep<Adapter>(scenario, reference, operation, "threads", "N_max_threads",
					&Config::N_max_threads, sweeps.N_max_threads, failed);
			}
		}

		// registers all supported operations of one library over the parameter matrix,
		// and records its version and capabilities in the output context
		template<typename Adapter>
		void register_library(const ComparisonMatrix& matrix, bool& failed)
		{
			const ComparisonModeValues& modes = matrix.modes;
			std::ostringstream capabilities;
			capabilities << "version=" << Adapter::version
				<< " rebuild=" << support_name(Adapter::rebuild_support)
				<< " query=" << support_name(Adapter::query_support)
				<< " query_batch=" << support_name(Adapter::batch_support)
				<< " parallel_rebuild=" << (Adapter::parallel_rebuild ? "yes" : "no")
				<< " batch_includes_rebuild=" << (Adapter::batch_includes_rebuild ? "yes" : "no");
			benchmark::AddCustomContext(std::string("library:") + Adapter::name, capabilities.str());

			// iterate over entire parameter matrix
			for (const Scenario* scenario : matrix.scenarios.values)
			for (int N_threads : modes.N_threads)
			for (auto query_placement : modes.query_placement)
			{
				Config reference = matrix.reference;
				reference.N_max_threads = N_threads;
				reference.query_placement = query_placement;

				for (const Operation& operation : matrix.operations.values)
				{
					if (!supports<Adapter>(operation)) { continue; }

					// register unused modes only with their first listed value, since other values measure the same work
					if (!uses_threads<Adapter>(operation) && N_threads != modes.N_threads.front()) { continue; }
					if (!uses_queries(operation) && query_placement != modes.query_placement.front()) { continue; }

					register_experiments<Adapter>(scenario, reference, operation, matrix.sweeps,
						matrix.queries_follow_points, failed);
				}
			}
		}
	}

	// the only list of compared libraries
	void register_benchmarks(const ComparisonMatrix& matrix, bool& failed)
	{
		register_library<CellmateAdapter<cellmate::PointAccess::cpy>>(matrix, failed);
		register_library<CellmateAdapter<cellmate::PointAccess::ref>>(matrix, failed);
		register_library<NanoflannAdapter>(matrix, failed);
		register_library<PicoTreeAdapter>(matrix, failed);
		register_library<UnibnOctreeAdapter>(matrix, failed);
		register_library<CompactNSearchAdapter>(matrix, failed);
		register_library<PoncaAdapter>(matrix, failed);
		register_library<CgalAdapter>(matrix, failed);
		register_library<BoostRtreeAdapter>(matrix, failed);
#ifdef CELLMATE_COMPARISON_TREENSEARCH
		register_library<TreeNSearchAdapter>(matrix, failed);
#endif
	}
}

int main(int argc, char** argv)
{
	const bool has_profile = argc > 1 && argv[1][0] != '-';
	const std::string profile = has_profile ? argv[1] : "comparison";
	cellmate_bench::ComparisonMatrix matrix;
	if (profile == "comparison") { matrix = cellmate_bench::comparison_matrix(); }
	else if (profile == "comp_points") { matrix = cellmate_bench::comp_points_matrix(); }
	else if (profile == "comp_queries") { matrix = cellmate_bench::comp_queries_matrix(); }
	else if (profile == "comp_neighbors") { matrix = cellmate_bench::comp_neighbors_matrix(); }
	else if (profile == "comp_threads") { matrix = cellmate_bench::comp_threads_matrix(); }
	else if (profile == "comp_scenarios") { matrix = cellmate_bench::comp_scenarios_matrix(); }
	else
	{
		std::cerr << "usage: cellmate_comparison [comparison|comp_points|comp_queries|comp_neighbors|comp_threads|comp_scenarios] [benchmark options]\n";
		return 1;
	}
	if (has_profile)
	{
		std::copy(argv + 2, argv + argc + 1, argv + 1);
		argc--;
	}

	// defaults before user arguments, so explicit flags and environment variables take precedence
	// (random interleaving spreads machine drift over all libraries instead of the last ones run)
	char repetitions[] = "--benchmark_repetitions=3";
	char min_time[] = "--benchmark_min_time=0.5s";
	char interleaving[] = "--benchmark_enable_random_interleaving=true";
	std::vector<char*> arguments{ argv[0] };
	if (!std::getenv("BENCHMARK_REPETITIONS")) { arguments.push_back(repetitions); }
	if (!std::getenv("BENCHMARK_MIN_TIME")) { arguments.push_back(min_time); }
	if (!std::getenv("BENCHMARK_ENABLE_RANDOM_INTERLEAVING")) { arguments.push_back(interleaving); }
	arguments.insert(arguments.end(), argv + 1, argv + argc);
	argc = int(arguments.size());
	arguments.push_back(nullptr);
	benchmark::Initialize(&argc, arguments.data());
	if (benchmark::ReportUnrecognizedArguments(argc, arguments.data())) { return 1; }
	benchmark::AddCustomContext("profile", profile);

	// record what is measured, since results are only comparable for the same commit, compiler and backend
	benchmark::AddCustomContext("cellmate_commit", CELLMATE_COMPARISON_COMMIT);
	benchmark::AddCustomContext("compiler", CELLMATE_COMPARISON_COMPILER);
	benchmark::AddCustomContext("build_config", CELLMATE_COMPARISON_CONFIG);
	benchmark::AddCustomContext("build_flags", CELLMATE_COMPARISON_FLAGS);
	benchmark::AddCustomContext("cellmate_simd", CELLMATE_DETAIL_SIMD_WIDTH == 512 ? "AVX-512"
		: CELLMATE_DETAIL_SIMD_WIDTH == 256 ? "AVX2" : "scalar");
#ifdef _OPENMP
	benchmark::AddCustomContext("openmp", std::to_string(_OPENMP));
#else
	benchmark::AddCustomContext("openmp", "off");
#endif
	bool failed = false;
	cellmate_bench::register_benchmarks(matrix, failed);
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
	return failed ? 1 : 0;
}
