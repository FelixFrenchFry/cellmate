#pragma once

#include "../comparison_adapter.hpp"

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/version.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

namespace cellmate_bench
{
	// Boost.Geometry R-tree with R* nodes of up to 16 entries, bulk loaded from a converted copy on each build
	class BoostRtreeAdapter
	{
		// points with their original index, indexed through the point of each pair
		using Point = boost::geometry::model::point<float, 3, boost::geometry::cs::cartesian>;
		using Box = boost::geometry::model::box<Point>;
		using Value = std::pair<Point, std::uint32_t>;
		using Tree = boost::geometry::index::rtree<Value, boost::geometry::index::rstar<16>>;

	public:

		static constexpr const char* name = "boost_rtree";
		static constexpr const char* version = BOOST_LIB_VERSION;
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::loop;
		static constexpr bool parallel_rebuild = false;
		static constexpr bool batch_includes_rebuild = false;
		static constexpr bool reorders_input = false;

		// native (point, index) pairs
		using QueryOutput = std::vector<Value>;

		// converts the positions into the point type outside timing
		BoostRtreeAdapter(const Workload& workload, const Config& config)
		{
			const cellmate_helpers::PointCloud& points = workload.points;
			points_.reserve(config.N_points);
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				points_.emplace_back(Point(points.x[i], points.y[i], points.z[i]), std::uint32_t(i));
			}
		}

		// constructs a new tree from the range, which bulk loads it with the packing algorithm
		void rebuild()
		{
			tree_.reset();
			tree_.emplace(points_.begin(), points_.end());
		}

		// queries the box around the sphere and filters by distance, the documented radius search,
		// with an inclusive cutoff (distance <= radius)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			output.clear();
			const Box box(Point(query_pos[0] - search_radius, query_pos[1] - search_radius, query_pos[2] - search_radius),
				Point(query_pos[0] + search_radius, query_pos[1] + search_radius, query_pos[2] + search_radius));
			const auto within_radius = [&query_pos](const Value& value)
			{
				const float dx = query_pos[0] - value.first.get<0>();
				const float dy = query_pos[1] - value.first.get<1>();
				const float dz = query_pos[2] - value.first.get<2>();
				return dx * dx + dy * dy + dz * dz <= search_radius * search_radius;
			};
			tree_->query(boost::geometry::index::intersects(box) && boost::geometry::index::satisfies(within_radius),
				std::back_inserter(output));
			return output.size();
		}

		void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors) const
		{
			for (const Value& match : output)
			{
				neighbors.push_back(match.second);
			}
		}

	private:

		std::vector<Value> points_;
		std::optional<Tree> tree_;
	};
}
