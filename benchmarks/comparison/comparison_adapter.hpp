#pragma once

#include "../bench_config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellmate_bench
{
	// how a library performs a benchmark operation
	enum class Support
	{
		none,   // not available
		native, // performed by the library's own API
		loop    // performed by the shared loop over single queries
	};

	inline const char* support_name(Support support)
	{
		switch (support)
		{
		case Support::native:
			return "native";
		case Support::loop:
			return "loop";
		default:
			return "none";
		}
	}

	// retained batch results, in the same layout as 'cellmate::NeighborSearch::query_batch()'
	// (the neighbors of query i are stored in [offsets[i], offsets[i + 1]))
	struct BatchResults
	{
		std::vector<std::uint32_t> neighbors;
		std::vector<std::size_t> offsets;
	};

	// reusable per-chunk storage of the shared loop
	template<typename Adapter>
	struct LoopScratch
	{
		std::vector<typename Adapter::QueryOutput> outputs;
		std::vector<std::vector<std::uint32_t>> neighbors;
		std::vector<std::size_t> chunk_offsets;
	};

	// answers a batch of queries with single queries, sequential version
	template<typename Adapter>
	void query_batch_loop_seq(const Adapter& adapter, const cellmate_helpers::PointCloud& queries,
		std::size_t N_queries, LoopScratch<Adapter>& scratch, BatchResults& results)
	{
		scratch.outputs.resize(1);
		typename Adapter::QueryOutput& output = scratch.outputs[0];
		results.neighbors.clear();
		results.offsets.resize(N_queries + 1);
		results.offsets[0] = 0;

		for (std::size_t i = 0; i < N_queries; i++)
		{
			adapter.query({ queries.x[i], queries.y[i], queries.z[i] }, output);
			Adapter::append_indices(output, results.neighbors);
			results.offsets[i + 1] = results.neighbors.size();
		}
	}

#ifdef _OPENMP
	// answers a batch of queries with single queries, parallel version
	// (one chunk of queries per thread, each with its own neighbor buffer, merged in query order)
	template<typename Adapter>
	void query_batch_loop_par(const Adapter& adapter, const cellmate_helpers::PointCloud& queries,
		std::size_t N_queries, int N_threads, LoopScratch<Adapter>& scratch, BatchResults& results)
	{
		const std::ptrdiff_t N_chunks = N_threads;
		scratch.outputs.resize(N_chunks);
		scratch.neighbors.resize(N_chunks);
		scratch.chunk_offsets.resize(N_chunks);
		results.offsets.resize(N_queries + 1);
		results.offsets[0] = 0;

		// parallel region for querying and merging
		#pragma omp parallel num_threads(N_threads)
		{
			// query each chunk, with offsets relative to its own buffer for now
			#pragma omp for schedule(static)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::size_t begin = N_queries * static_cast<std::size_t>(chunk) / static_cast<std::size_t>(N_chunks);
				const std::size_t end = N_queries * static_cast<std::size_t>(chunk + 1) / static_cast<std::size_t>(N_chunks);
				typename Adapter::QueryOutput& output = scratch.outputs[chunk];
				std::vector<std::uint32_t>& neighbors = scratch.neighbors[chunk];
				neighbors.clear();
				for (std::size_t i = begin; i < end; i++)
				{
					adapter.query({ queries.x[i], queries.y[i], queries.z[i] }, output);
					Adapter::append_indices(output, neighbors);
					results.offsets[i + 1] = neighbors.size();
				}
			}

			// convert buffer sizes into starting positions in the flat neighbor array
			#pragma omp single
			{
				std::size_t offset = 0;
				for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
				{
					scratch.chunk_offsets[chunk] = offset;
					offset += scratch.neighbors[chunk].size();
				}
				results.neighbors.resize(offset);
			}

			// shift the offsets of each chunk behind the preceding buffers and copy its neighbors into place
			#pragma omp for schedule(static)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::size_t begin = N_queries * static_cast<std::size_t>(chunk) / static_cast<std::size_t>(N_chunks);
				const std::size_t end = N_queries * static_cast<std::size_t>(chunk + 1) / static_cast<std::size_t>(N_chunks);
				const std::size_t offset = scratch.chunk_offsets[chunk];
				for (std::size_t i = begin; i < end; i++)
				{
					results.offsets[i + 1] += offset;
				}
				std::copy(scratch.neighbors[chunk].begin(), scratch.neighbors[chunk].end(),
					results.neighbors.begin() + offset);
			}
		}
	}
#endif

	// selects the sequential or parallel version of the shared loop on the execution mode
	template<typename Adapter>
	void query_batch_loop(const Adapter& adapter, const cellmate_helpers::PointCloud& queries,
		std::size_t N_queries, cellmate::ComputeMode exec, int N_threads,
		LoopScratch<Adapter>& scratch, BatchResults& results)
	{
#ifdef _OPENMP
		if (exec == cellmate::ComputeMode::par)
		{
			query_batch_loop_par(adapter, queries, N_queries, N_threads, scratch, results);
		}
		else
		{
			query_batch_loop_seq(adapter, queries, N_queries, scratch, results);
		}
#else
		(void)exec;
		(void)N_threads;
		query_batch_loop_seq(adapter, queries, N_queries, scratch, results);
#endif
	}
}
