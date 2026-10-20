#pragma once

#include "../comparison_adapter.hpp"

#include <CompactNSearch>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cellmate_bench
{
	// CompactNSearch, z-sorting a copy of the positions on each rebuild as the library recommends
	// (results refer to the z-sorted order and are only mapped back to original indices for validation)
	class CompactNSearchAdapter
	{
		using Points = std::vector<std::array<float, 3>>;

	public:

		static constexpr const char* name = "compact_nsearch";
		static constexpr const char* version = "b40afcf47fe1963b363eba2371f04b42720fcb1d";
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::native;
		static constexpr bool parallel_rebuild = true;
		static constexpr bool batch_includes_rebuild = false;
		static constexpr bool reorders_input = true;

		// native neighbor lists, one per point set
		using QueryOutput = std::vector<std::vector<unsigned int>>;

		// converts the positions into the interleaved layout outside timing
		CompactNSearchAdapter(const Workload& workload, const Config& config)
			: unsorted_points_(config.N_points), points_(config.N_points), original_indices_(config.N_points),
			self_search_(workload.queries.x == workload.points.x && workload.queries.y == workload.points.y &&
				workload.queries.z == workload.points.z),
			search_(search_radius)
		{
			const cellmate_helpers::PointCloud& points = workload.points;
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				unsorted_points_[i] = { points.x[i], points.y[i], points.z[i] };
			}

			// the points are found by queries, and only search themselves when they are the queries
			search_.add_point_set(points_[0].data(), config.N_points, true, self_search_, true);

			// interleaved query positions for batches, registered as a point set on the first batch
			const cellmate_helpers::PointCloud& queries = workload.queries;
			queries_.resize(config.N_queries);
			for (std::size_t i = 0; i < config.N_queries; i++)
			{
				queries_[i] = { queries.x[i], queries.y[i], queries.z[i] };
			}
		}

		// restores the unsorted positions, called outside timing before each rebuild
		void prepare_rebuild()
		{
			std::copy(unsorted_points_.begin(), unsorted_points_.end(), points_.begin());
		}

		// z-sorts the positions and an index array tracking their original indices, then forces a complete
		// reinitialization of the hash table, instead of an incremental update
		void rebuild()
		{
			search_.z_sort();
			search_.point_set(points_set).sort_field(points_.data());
			std::iota(original_indices_.begin(), original_indices_.end(), 0u);
			search_.point_set(points_set).sort_field(original_indices_.data());
			search_.reset();
			search_.update_point_sets();
		}

		// takes the plain radius, with a strict cutoff (distance < radius)
		// (not const, since the library's query is not, although it only reads the hash table)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output)
		{
			search_.find_neighbors(query_pos.data(), output);
			return output[points_set].size();
		}

		// searches all queries at once, within the point set when the queries are the points (the native use),
		// otherwise after registering them as a separate point set on the first call
		// (searches within one set exclude the point itself, which is added back outside timing)
		void query_batch(std::size_t N_queries, BatchResults&)
		{
			if (N_queries != queries_.size())
			{
				throw std::invalid_argument("compact_nsearch batches always contain all queries");
			}
			if (self_search_)
			{
				search_.find_neighbors(false);
			}
			else if (search_.n_point_sets() == 1)
			{
				// queries only search the points (not each other), changing the activation forces a reinitialization
				search_.add_point_set(queries_[0].data(), queries_.size(), true, true, false);
				search_.set_active(queries_set, queries_set, false);
				search_.find_neighbors(true);
			}
			else
			{
				search_.find_neighbors(false);
			}
		}

		std::size_t batch_neighbor_count(const BatchResults&) const
		{
			const CompactNSearch::PointSet& queries = search_.point_set(self_search_ ? points_set : queries_set);
			std::size_t N_neighbors = self_search_ ? queries.n_points() : 0;
			for (unsigned int i = 0; i < queries.n_points(); i++)
			{
				N_neighbors += queries.n_neighbors(points_set, i);
			}
			return N_neighbors;
		}

		void batch_neighbors(const BatchResults&, std::size_t query, std::vector<std::uint32_t>& neighbors) const
		{
			neighbors.clear();
			const unsigned int set = self_search_ ? points_set : queries_set;
			const unsigned int index = self_search_ ? unsigned(std::find(original_indices_.begin(),
				original_indices_.end(), std::uint32_t(query)) - original_indices_.begin()) : unsigned(query);
			for (unsigned int sorted : search_.point_set(set).neighbor_list(points_set, index))
			{
				neighbors.push_back(original_indices_[sorted]);
			}
			if (self_search_)
			{
				neighbors.push_back(std::uint32_t(query));
			}
		}

		void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors) const
		{
			for (unsigned int sorted : output[points_set])
			{
				neighbors.push_back(original_indices_[sorted]);
			}
		}

	private:

		static constexpr unsigned int points_set = 0;
		static constexpr unsigned int queries_set = 1;

		Points unsorted_points_;
		Points points_;
		Points queries_;
		std::vector<std::uint32_t> original_indices_;
		bool self_search_;
		CompactNSearch::NeighborhoodSearch search_;
	};
}
