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

		using QueryOutput = std::vector<std::uint32_t>;

		CellmateAdapter(const cellmate_helpers::PointCloud& points, const Config& config)
			: points_(points), N_points_(config.N_points), compute_mode_(config.compute_mode),
			search_(search_radius, point_access)
		{
		}

		void rebuild()
		{
			search_.rebuild(points_.x.data(), points_.y.data(), points_.z.data(), N_points_, compute_mode_);
		}

		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			search_.query(query_pos, output);
			return output.size();
		}

		void query_batch(const cellmate_helpers::PointCloud& queries, std::size_t N_queries, BatchResults& results) const
		{
			search_.query_batch(queries.x.data(), queries.y.data(), queries.z.data(), N_queries,
				results.neighbors, results.offsets, compute_mode_, cellmate::QueryOrder::morton);
		}

		static void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors)
		{
			neighbors.insert(neighbors.end(), output.begin(), output.end());
		}

	private:

		const cellmate_helpers::PointCloud& points_;
		std::size_t N_points_;
		cellmate::ComputeMode compute_mode_;
		cellmate::NeighborSearch search_;
	};
}
