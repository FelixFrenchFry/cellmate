#pragma once

#include "../comparison_adapter.hpp"

#include <cellmate/neighbor_search.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellmate_bench
{
	// cellmate with copied or referenced positions, using the default query order
	template<cellmate::PointAccess point_access>
	class CellmateAdapter
	{
	public:

		static constexpr const char* name = point_access == cellmate::PointAccess::cpy ? "cellmate_cpy" : "cellmate_ref";
		static constexpr const char* version = "this repository";
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::native;
		static constexpr bool parallel_rebuild = true;
		static constexpr bool batch_includes_rebuild = false;
		static constexpr bool reorders_input = false;

		// native indices
		using QueryOutput = std::vector<std::uint32_t>;

		// uses the positions without conversion, copying them on each rebuild only with 'cpy'
		CellmateAdapter(const Workload& workload, const Config& config)
			: workload_(workload), N_points_(config.N_points), compute_mode_(config.compute_mode),
			search_(search_radius, point_access)
		{
		}

		// rebuilds the existing search object, reusing its storage
		void rebuild()
		{
			const cellmate_helpers::PointCloud& points = workload_.points;
			search_.rebuild(points.x.data(), points.y.data(), points.z.data(), N_points_, compute_mode_);
		}

		// takes the plain radius, with an inclusive cutoff (distance <= radius)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			search_.query(query_pos, output);
			return output.size();
		}

		void query_batch(std::size_t N_queries, BatchResults& results)
		{
			const cellmate_helpers::PointCloud& queries = workload_.queries;
			search_.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), N_queries,
				results.neighbors, results.offsets, compute_mode_, cellmate::QueryOrder::morton);
		}

		std::size_t batch_neighbor_count(const BatchResults& results) const
		{
			return cellmate_bench::batch_neighbor_count(results);
		}

		void batch_neighbors(const BatchResults& results, std::size_t query, std::vector<std::uint32_t>& neighbors) const
		{
			cellmate_bench::batch_neighbors(results, query, neighbors);
		}

		void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors) const
		{
			neighbors.insert(neighbors.end(), output.begin(), output.end());
		}

	private:

		const Workload& workload_;
		std::size_t N_points_;
		cellmate::ComputeMode compute_mode_;
		cellmate::NeighborSearch search_;
	};
}
