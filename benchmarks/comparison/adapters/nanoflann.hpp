#pragma once

#include "../comparison_adapter.hpp"

#include <nanoflann.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellmate_bench
{
	// nanoflann k-d tree with library default parameters, reading the positions in place
	class NanoflannAdapter
	{
		// dataset adaptor over the separate (x, y, z) arrays of the benchmark
		struct Dataset
		{
			const cellmate_helpers::PointCloud* points;
			std::size_t N_points;

			std::size_t kdtree_get_point_count() const { return N_points; }

			float kdtree_get_pt(std::uint32_t idx, std::size_t dim) const
			{
				return dim == 0 ? points->x[idx] : dim == 1 ? points->y[idx] : points->z[idx];
			}

			// no precomputed bounding box, let nanoflann compute it
			template<typename BoundingBox>
			bool kdtree_get_bbox(BoundingBox&) const { return false; }
		};

		using Index = nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, Dataset, float, std::uint32_t>, Dataset, 3, std::uint32_t>;

	public:

		static constexpr const char* name = "nanoflann";
		static constexpr const char* version = NANOFLANN_VERSION_STRING;
		static constexpr Support rebuild_support = Support::native;
		static constexpr Support query_support = Support::native;
		static constexpr Support batch_support = Support::loop;
		static constexpr bool parallel_rebuild = true;

		// native (index, squared distance) pairs
		using QueryOutput = std::vector<nanoflann::ResultItem<std::uint32_t, float>>;

		// builds use 'std::async' threads, queries leave the results unsorted (sorted by default)
		NanoflannAdapter(const cellmate_helpers::PointCloud& points, const Config& config)
			: dataset_{ &points, config.N_points },
			index_(3, dataset_, nanoflann::KDTreeSingleIndexAdaptorParams(10, // library default leaf size
				nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex, unsigned(config.N_max_threads)))
		{
		}

		void rebuild()
		{
			index_.buildIndex();
		}

		// takes the squared radius, with a strict cutoff (distance < radius)
		std::size_t query(std::array<float, 3> query_pos, QueryOutput& output) const
		{
			return index_.radiusSearch(query_pos.data(), search_radius * search_radius,
				output, nanoflann::SearchParameters(0, false));
		}

		static void append_indices(const QueryOutput& output, std::vector<std::uint32_t>& neighbors)
		{
			for (const auto& match : output)
			{
				neighbors.push_back(match.first);
			}
		}

	private:

		Dataset dataset_;
		Index index_;
	};
}
