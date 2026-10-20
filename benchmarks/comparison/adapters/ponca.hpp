#pragma once

#include "../comparison_adapter.hpp"

#include <Eigen/Core>
#include <Ponca/SpatialPartitioning>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellmate_bench
{
	// Ponca k-d tree with library default parameters, copying the points into the tree on each build
	class PoncaAdapter
	{
		// point type with the 'Dim', 'Scalar', 'VectorType' and 'pos()' members expected by the tree
		struct Point
		{
			enum { Dim = 3 };
			using Scalar = float;
			using VectorType = Eigen::Matrix<float, 3, 1>;

			Point(const VectorType& pos = VectorType::Zero()) : pos_(pos) {}
			const VectorType& pos() const { return pos_; }

			VectorType pos_;
		};

		using Tree = Ponca::KdTreeDense<Point>;

	public:

		static constexpr const char* name = "ponca";
		static constexpr const char* version = "1.4";
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::loop;
		static constexpr bool parallel_rebuild = false;
		static constexpr bool batch_includes_rebuild = false;
		static constexpr bool reorders_input = false;

		// indices collected from the lazy native query
		using QueryOutput = std::vector<int>;

		// converts the positions into the point type outside timing
		PoncaAdapter(const Workload& workload, const Config& config)
		{
			const cellmate_helpers::PointCloud& points = workload.points;
			points_.reserve(config.N_points);
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				points_.emplace_back(Point::VectorType(points.x[i], points.y[i], points.z[i]));
			}
		}

		// copies the points into the tree, which clears the previous one first
		void rebuild()
		{
			tree_.build(points_);
		}

		// takes the plain radius, with a strict cutoff (distance < radius)
		// (each query object holds its own traversal stack, only reading the tree)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			output.clear();
			for (int index : tree_.range_neighbors(Point::VectorType(query_pos[0], query_pos[1], query_pos[2]), search_radius))
			{
				output.push_back(index);
			}
			return output.size();
		}

		void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors) const
		{
			for (int index : output)
			{
				neighbors.push_back(static_cast<std::uint32_t>(index));
			}
		}

	private:

		std::vector<Point> points_;
		Tree tree_;
	};
}
