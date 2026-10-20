#pragma once

#include "../comparison_adapter.hpp"

#include <TreeNSearch.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cellmate_bench
{
	// TreeNSearch with library default parameters, z-sorting a copy of the positions on each rebuild as the library recommends
	// (every batch builds the octree and searches in one 'run()', so batches are only measured with rebuilds,
	// and results refer to the z-sorted order and are only mapped back to original indices for validation)
	class TreeNSearchAdapter
	{
		using Points = std::vector<std::array<float, 3>>;

	public:

		static constexpr const char* name = "tree_nsearch";
		static constexpr const char* version = "61cf111f65c01f1d71fb5023ac72ce6bb0e8d6f9";
		static constexpr Support rebuild_support = Support::none;
		static constexpr Support query_support = Support::none;
		static constexpr Support batch_support = Support::native;
		static constexpr bool parallel_rebuild = true;
		static constexpr bool batch_includes_rebuild = true;
		static constexpr bool reorders_input = true;

		// unused, since there are no single queries
		using QueryOutput = std::vector<std::uint32_t>;

		// converts the positions into the interleaved layout outside timing, with points that search themselves
		// when they are the queries (the native use), otherwise with a query set that only searches the points
		TreeNSearchAdapter(const Workload& workload, const Config& config)
			: unsorted_points_(config.N_points), points_{ Points(config.N_points), Points(config.N_points) },
			queries_(config.N_queries), original_indices_(config.N_points),
			self_search_(workload.queries.x == workload.points.x && workload.queries.y == workload.points.y &&
				workload.queries.z == workload.points.z)
		{
			const cellmate_helpers::PointCloud& points = workload.points;
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				unsorted_points_[i] = { points.x[i], points.y[i], points.z[i] };
			}
			const cellmate_helpers::PointCloud& queries = workload.queries;
			for (std::size_t i = 0; i < config.N_queries; i++)
			{
				queries_[i] = { queries.x[i], queries.y[i], queries.z[i] };
			}

			search_.set_search_radius(search_radius);
			search_.set_n_threads(config.N_max_threads);
			search_.add_point_set(points_[current_][0].data(), int(config.N_points));
			if (self_search_)
			{
				search_.set_active_search(points_set, points_set, true);
			}
			else
			{
				search_.add_point_set(queries_[0].data(), int(config.N_queries));
				search_.set_all_searches(false);
				search_.set_active_search(queries_set, points_set, true);
			}
		}

		// restores the unsorted positions into the other buffer, called outside timing before each rebuild
		// (the new address invalidates the previous octree, which the z-sort would otherwise reuse)
		void prepare_rebuild()
		{
			current_ = 1 - current_;
			std::copy(unsorted_points_.begin(), unsorted_points_.end(), points_[current_].begin());
			search_.resize_point_set(points_set, points_[current_][0].data(), int(unsorted_points_.size()));
		}

		// z-sorts the positions and an index array tracking their original indices,
		// since every batch builds the octree itself
		void rebuild()
		{
			search_.prepare_zsort();
			search_.apply_zsort(points_set, points_[current_][0].data(), 3);
			std::iota(original_indices_.begin(), original_indices_.end(), 0u);
			search_.apply_zsort(points_set, original_indices_.data(), 1);
		}

		// builds the octree and searches all queries at once, with an inclusive cutoff (distance <= radius)
		// (searches within one set exclude the point itself, which is added back outside timing)
		void query_batch(std::size_t N_queries, BatchResults&)
		{
			if (N_queries != queries_.size())
			{
				throw std::invalid_argument("tree_nsearch batches always contain all queries");
			}
			search_.run();
		}

		std::size_t batch_neighbor_count(const BatchResults&) const
		{
			const int set = self_search_ ? points_set : queries_set;
			std::size_t N_neighbors = self_search_ ? queries_.size() : 0;
			for (std::size_t i = 0; i < queries_.size(); i++)
			{
				N_neighbors += std::size_t(search_.get_neighborlist(set, points_set, int(i)).size());
			}
			return N_neighbors;
		}

		void batch_neighbors(const BatchResults&, std::size_t query, std::vector<std::uint32_t>& neighbors) const
		{
			const int set = self_search_ ? points_set : queries_set;
			const int index = self_search_ ? int(std::find(original_indices_.begin(), original_indices_.end(),
				std::uint32_t(query)) - original_indices_.begin()) : int(query);
			const tns::NeighborList list = search_.get_neighborlist(set, points_set, index);
			neighbors.clear();
			for (int k = 0; k < list.size(); k++)
			{
				neighbors.push_back(original_indices_[list[k]]);
			}
			if (self_search_)
			{
				neighbors.push_back(std::uint32_t(query));
			}
		}

	private:

		static constexpr int points_set = 0;
		static constexpr int queries_set = 1;

		Points unsorted_points_;
		std::array<Points, 2> points_;
		std::size_t current_ = 0;
		Points queries_;
		std::vector<std::uint32_t> original_indices_;
		bool self_search_;
		tns::TreeNSearch search_;
	};
}
