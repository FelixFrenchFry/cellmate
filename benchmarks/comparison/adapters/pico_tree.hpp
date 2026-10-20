#pragma once

#include "../comparison_adapter.hpp"

#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace cellmate_bench
{
	// PicoTree k-d tree with the leaf size of its own benchmarks, reading a converted copy in place
	class PicoTreeAdapter
	{
		using Points = std::vector<std::array<float, 3>>;
		using Tree = pico_tree::kd_tree<std::reference_wrapper<Points>>;

	public:

		static constexpr const char* name = "pico_tree";
		static constexpr const char* version = "1.0.0";
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::loop;
		static constexpr bool parallel_rebuild = false;
		static constexpr bool batch_includes_rebuild = false;
		static constexpr bool reorders_input = false;

		// native (index, squared distance) pairs
		using QueryOutput = std::vector<Tree::neighbor_type>;

		// converts the positions into the interleaved layout outside timing
		PicoTreeAdapter(const Workload& workload, const Config& config)
			: points_(config.N_points)
		{
			const cellmate_helpers::PointCloud& points = workload.points;
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				points_[i] = { points.x[i], points.y[i], points.z[i] };
			}
		}

		// constructs a new tree, since there is no rebuild of an existing one
		void rebuild()
		{
			tree_.reset();
			tree_.emplace(std::ref(points_), pico_tree::max_leaf_size_t(10));
		}

		// takes the squared radius, with a strict cutoff (distance < radius)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			tree_->search_radius(query_pos, search_radius * search_radius, output);
			return output.size();
		}

		void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors) const
		{
			for (const auto& match : output)
			{
				neighbors.push_back(static_cast<std::uint32_t>(match.index));
			}
		}

	private:

		Points points_;
		std::optional<Tree> tree_;
	};
}
