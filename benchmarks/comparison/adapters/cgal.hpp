#pragma once

#include "../comparison_adapter.hpp"

#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/version.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace cellmate_bench
{
	// CGAL k-d tree with library default parameters on float coordinates, copying the points into the tree on each build
	class CgalAdapter
	{
		// points with their original index, searched through the point of each pair
		using Kernel = CGAL::Simple_cartesian<float>;
		using Point = std::pair<Kernel::Point_3, std::uint32_t>;
		using Traits = CGAL::Search_traits_adapter<Point, CGAL::First_of_pair_property_map<Point>,
			CGAL::Search_traits_3<Kernel>>;
		using Tree = CGAL::Kd_tree<Traits>;
		using Sphere = CGAL::Fuzzy_sphere<Traits>;

	public:

		static constexpr const char* name = "cgal";
		static constexpr const char* version = CGAL_VERSION_STR;
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::loop;
		static constexpr bool parallel_rebuild = false;
		static constexpr bool batch_includes_rebuild = false;
		static constexpr bool reorders_input = false;

		// native (point, index) pairs
		using QueryOutput = std::vector<Point>;

		// converts the positions into the point type outside timing
		CgalAdapter(const Workload& workload, const Config& config)
		{
			const cellmate_helpers::PointCloud& points = workload.points;
			points_.reserve(config.N_points);
			for (std::size_t i = 0; i < config.N_points; i++)
			{
				points_.emplace_back(Kernel::Point_3(points.x[i], points.y[i], points.z[i]), std::uint32_t(i));
			}
		}

		// copies the points into the existing tree and builds it explicitly,
		// since queries would otherwise build it lazily on the first search
		void rebuild()
		{
			tree_.clear();
			tree_.insert(points_.begin(), points_.end());
			tree_.build();
		}

		// takes the plain radius and no fuzziness, with an inclusive cutoff (distance <= radius)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			output.clear();
			const Sphere sphere(Kernel::Point_3(query_pos[0], query_pos[1], query_pos[2]), search_radius, 0.0f);
			tree_.search(std::back_inserter(output), sphere);
			return output.size();
		}

		void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors) const
		{
			for (const Point& match : output)
			{
				neighbors.push_back(match.second);
			}
		}

	private:

		std::vector<Point> points_;
		Tree tree_;
	};
}
