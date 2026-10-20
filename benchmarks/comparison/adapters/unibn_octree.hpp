#pragma once

#include "../comparison_adapter.hpp"

#include <Octree.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellmate_bench
{
	// unibn::Octree with library default parameters, reading a converted copy in place
	class UnibnOctreeAdapter
	{
		// point type with the public x, y and z members expected by the default access traits
		struct Point
		{
			float x, y, z;
		};

		using Octree = unibn::Octree<Point>;

	public:

		static constexpr const char* name = "unibn_octree";
		static constexpr const char* version = "8e3927d48d5ce61f94aab6090d77b4434f87dc89";
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::loop;
		static constexpr bool parallel_rebuild = false;
		static constexpr bool batch_includes_rebuild = false;
		static constexpr bool reorders_input = false;

		// native indices
		using QueryOutput = std::vector<std::uint32_t>;

		// converts the positions into the interleaved layout outside timing
		UnibnOctreeAdapter(const Workload& workload, const Config& config)
			: points_(config.N_points)
		{
			const cellmate_helpers::PointCloud& points = workload.points;
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				points_[i] = { points.x[i], points.y[i], points.z[i] };
			}
		}

		// clears the previous octree before building the new one
		void rebuild()
		{
			octree_.initialize(points_);
		}

		// takes the plain radius, with a strict cutoff (distance < radius)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			octree_.radiusNeighbors<unibn::L2Distance<Point>>({ query_pos[0], query_pos[1], query_pos[2] },
				search_radius, output);
			return output.size();
		}

		void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors) const
		{
			neighbors.insert(neighbors.end(), output.begin(), output.end());
		}

	private:

		std::vector<Point> points_;
		Octree octree_;
	};
}
