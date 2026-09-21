#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cellmate
{
	using std::uint32_t;
	using std::uint64_t;

	enum class Exec
	{
		seq,
		par
	};

	class NeighborSearch
	{
	public:

		explicit NeighborSearch(float radius) : radius_(radius)
		{
			if (!std::isfinite(radius) || radius <= 0)
			{
				throw std::invalid_argument("cellmate::NeighborSearch() radius must be finite and positive");
			}
		}

		// TODO: XML documentation?
		void rebuild(const float* x, const float* y, const float* z, std::size_t N_points, Exec exec = Exec::par)
		{
			// clear previous state
			positions_ = {};
			point_order_.clear();
			cell_keys_.clear();
			cell_starts_.clear();
			cell_cache_.fill(CellCacheEntry{});

			// validate input
			if (N_points == 0) { return; }
			if (!x || !y || !z)
			{
				throw std::invalid_argument("cellmate::rebuild() requires three coordinate arrays");
			}

#ifdef _OPENMP
			if (exec == Exec::par &&
				N_points <= static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
			{
				rebuild_par(x, y, z, static_cast<std::ptrdiff_t>(N_points));
			}
			else
			{
				rebuild_seq(x, y, z, N_points);
			}
#else
			(void)exec;
			rebuild_seq(x, y, z, N_points);
#endif

			positions_ = { x, y, z };
		}

		// TODO: XML documentation?
		// WARNING: concurrent queries on the same object not supported yet! only one thread at a time!
		void find_neighbors(std::array<float, 3> query_pos, std::vector<std::size_t>& neighbors) const
		{
			neighbors.clear();

			// validate input
			if (!positions_[0]) { return; }
			if (!std::isfinite(query_pos[0]) ||
				!std::isfinite(query_pos[1]) ||
				!std::isfinite(query_pos[2]))
			{
				throw std::invalid_argument("cellmate::find_neighbors() query position must be finite");
			}

			// query position relative to the grid in cell units
			const double center_x = (double(query_pos[0]) - origin_[0]) / radius_;
			const double center_y = (double(query_pos[1]) - origin_[1]) / radius_;
			const double center_z = (double(query_pos[2]) - origin_[2]) / radius_;

			// expand cell bounds to account for fp rounding
			// (affects which canididate cells are included in the search, not the actual distance check)
			const double margin_x = 8 * std::numeric_limits<double>::epsilon() * (std::abs(center_x) + 1);
			const double margin_y = 8 * std::numeric_limits<double>::epsilon() * (std::abs(center_y) + 1);
			const double margin_z = 8 * std::numeric_limits<double>::epsilon() * (std::abs(center_z) + 1);

			// find candidate cell bounds, clipped to the extent of the grid
			const double first_x = (std::max)(0.0, std::floor(center_x - 1 - margin_x));
			const double first_y = (std::max)(0.0, std::floor(center_y - 1 - margin_y));
			const double first_z = (std::max)(0.0, std::floor(center_z - 1 - margin_z));
			const double last_x = (std::min)(double(last_cell_[0]), std::floor(center_x + 1 + margin_x));
			const double last_y = (std::min)(double(last_cell_[1]), std::floor(center_y + 1 + margin_y));
			const double last_z = (std::min)(double(last_cell_[2]), std::floor(center_z + 1 + margin_z));
			if (first_x > last_x || first_y > last_y || first_z > last_z) { return; }

			const std::array<uint32_t, 3> begin{ static_cast<uint32_t>(first_x),
												 static_cast<uint32_t>(first_y),
												 static_cast<uint32_t>(first_z) };
			const std::array<uint32_t, 3> end{ static_cast<uint32_t>(last_x),
											   static_cast<uint32_t>(last_y),
											   static_cast<uint32_t>(last_z) };

			// visit candidate cells and check contained points
			for (uint32_t z = begin[2]; z <= end[2]; z++)
			{
				const uint64_t z_code = spread_bits(z) << 2;
				for (uint32_t y = begin[1]; y <= end[1]; y++)
				{
					const uint64_t yz_code = z_code | (spread_bits(y) << 1);
					for (uint32_t x = begin[0]; x <= end[0]; x++)
					{
						const uint64_t key = yz_code | spread_bits(x);
						const std::size_t cell_idx = find_cell(key);
						if (cell_idx == cell_keys_.size()) { continue; }

						// collect original indices of points within the search radius
						for (std::size_t i = cell_starts_[cell_idx]; i < cell_starts_[cell_idx + 1]; i++)
						{
							const std::size_t idx = point_order_[i].idx;
							const double dx = double(query_pos[0]) - positions_[0][idx];
							const double dy = double(query_pos[1]) - positions_[1][idx];
							const double dz = double(query_pos[2]) - positions_[2][idx];

							if (dx * dx + dy * dy + dz * dz <= radius_ * radius_)
							{
								neighbors.push_back(idx);
							}
						}
					}
				}
			}
		}

	private:

		// TODO: go without struct?
		struct PointEntry
		{
			uint64_t key;
			std::size_t idx;
		};

		struct CellCacheEntry
		{
			// invalid Morton key marks an unused entry
			uint64_t key = (std::numeric_limits<uint64_t>::max)();
			std::size_t idx = 0;
		};

		double radius_;
		std::array<const float*, 3> positions_{};
		std::array<float, 3> origin_{};
		std::array<uint32_t, 3> last_cell_{};
		std::vector<PointEntry> point_order_;
		std::vector<uint64_t> cell_keys_;
		std::vector<std::size_t> cell_starts_;

		// cell lookup cache stores cell indices and empty-cell results
		mutable std::array<CellCacheEntry, 256> cell_cache_{};

		void rebuild_seq(const float* x, const float* y, const float* z, std::size_t N_points)
		{
			// compute minimum and maximum (x,y,z)
			std::array<float, 3> minimum{ x[0], y[0], z[0] };
			std::array<float, 3> maximum = minimum;
			for (std::size_t i = 0; i < N_points; i++)
			{
				if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]))
				{
					throw std::invalid_argument("cellmate::rebuild() coordinates must be finite");
				}

				minimum[0] = (std::min)(minimum[0], x[i]);
				minimum[1] = (std::min)(minimum[1], y[i]);
				minimum[2] = (std::min)(minimum[2], z[i]);
				maximum[0] = (std::max)(maximum[0], x[i]);
				maximum[1] = (std::max)(maximum[1], y[i]);
				maximum[2] = (std::max)(maximum[2], z[i]);
			}

			// set grid origin at the minimum (x,y,z) to avoid negative Morton keys and validate grid extent
			origin_ = minimum;
			last_cell_ = { cell_coord(maximum[0], 0), cell_coord(maximum[1], 1), cell_coord(maximum[2], 2) };

			// assign Morton keys and original indices to each point
			point_order_.resize(N_points); // TODO: find alternative that grows with extra margin to avoid repeated allocations?
			for (std::size_t i = 0; i < N_points; i++)
			{
				const uint32_t cx = cell_coord(x[i], 0);
				const uint32_t cy = cell_coord(y[i], 1);
				const uint32_t cz = cell_coord(z[i], 2);
				point_order_[i] = { morton_key(cx, cy, cz), i };
			}

			sort_point_order();
			build_cell_table(N_points);
		}

#ifdef _OPENMP
		void rebuild_par(const float* x, const float* y, const float* z, std::ptrdiff_t N_points)
		{
			// compute minimum and maximum (x,y,z)
			// (compute local bounds per thread, then merge into grid bounds)
			std::array<float, 3> minimum{ x[0], y[0], z[0] };
			std::array<float, 3> maximum = minimum;
			bool invalid = false;
			#pragma omp parallel
			{
				std::array<float, 3> local_min{ x[0], y[0], z[0] };
				std::array<float, 3> local_max = local_min;
				bool local_invalid = false;

				#pragma omp for schedule(static)
				for (std::ptrdiff_t i = 0; i < N_points; i++)
				{
					if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]))
					{
						local_invalid = true;
						continue;
					}

					local_min[0] = (std::min)(local_min[0], x[i]);
					local_min[1] = (std::min)(local_min[1], y[i]);
					local_min[2] = (std::min)(local_min[2], z[i]);
					local_max[0] = (std::max)(local_max[0], x[i]);
					local_max[1] = (std::max)(local_max[1], y[i]);
					local_max[2] = (std::max)(local_max[2], z[i]);
				}

				#pragma omp critical(cellmate_rebuild_bounds)
				{
					minimum[0] = (std::min)(minimum[0], local_min[0]);
					minimum[1] = (std::min)(minimum[1], local_min[1]);
					minimum[2] = (std::min)(minimum[2], local_min[2]);
					maximum[0] = (std::max)(maximum[0], local_max[0]);
					maximum[1] = (std::max)(maximum[1], local_max[1]);
					maximum[2] = (std::max)(maximum[2], local_max[2]);
					invalid = invalid || local_invalid;
				}
			}

			if (invalid)
			{
				throw std::invalid_argument("cellmate::rebuild() coordinates must be finite");
			}

			// set grid origin at the minimum (x,y,z) to avoid negative Morton keys and validate grid extent
			origin_ = minimum;
			last_cell_ = { cell_coord(maximum[0], 0), cell_coord(maximum[1], 1), cell_coord(maximum[2], 2) };

			// assign Morton keys and original indices to each point in parallel
			point_order_.resize(N_points);
			#pragma omp parallel for schedule(static)
			for (std::ptrdiff_t i = 0; i < N_points; i++)
			{
				const uint32_t cx = cell_coord(x[i], 0);
				const uint32_t cy = cell_coord(y[i], 1);
				const uint32_t cz = cell_coord(z[i], 2);
				point_order_[i] = { morton_key(cx, cy, cz), static_cast<std::size_t>(i) };
			}

			sort_point_order();
			build_cell_table(N_points);
		}
#endif

		// sort point indices by Morton key
		void sort_point_order()
		{
			// TODO: use faster sorting algo?
			std::sort(point_order_.begin(), point_order_.end(),
				[](const PointEntry& a, const PointEntry& b)
				{
					return a.key < b.key || (a.key == b.key && a.idx < b.idx);
				});
		}

		// store each occupied cell's key and starting point offset
		void build_cell_table(std::size_t N_points)
		{
			std::size_t begin = 0;
			for (std::size_t end = 1; end <= N_points; end++)
			{
				if (end == N_points || point_order_[end].key != point_order_[begin].key)
				{
					cell_keys_.push_back(point_order_[begin].key);
					cell_starts_.push_back(begin);
					begin = end;
				}
			}

			cell_starts_.push_back(N_points); // end of the last cell
		}

		// reuse the cached result if its key matches, otherwise search and replace it
		std::size_t find_cell(uint64_t key) const
		{
			const std::size_t slot = static_cast<std::size_t>(key ^ (key >> 32)) & (cell_cache_.size() - 1);
			CellCacheEntry& entry = cell_cache_[slot];
			if (entry.key == key) { return entry.idx; }

			const std::vector<uint64_t>::const_iterator cell = std::lower_bound(cell_keys_.begin(), cell_keys_.end(), key);
			entry.key = key;
			entry.idx = cell == cell_keys_.end() || *cell != key
				? cell_keys_.size()
				: static_cast<std::size_t>(cell - cell_keys_.begin());
			
			return entry.idx;
		}

		// convert a position to a cell coordinate relative to the grid origin
		uint32_t cell_coord(float p, std::size_t axis) const
		{
			const double coordinate = std::floor((double(p) - origin_[axis]) / radius_);

			if (coordinate >= (1u << 21)) // TODO: disable in release builds?
			{
				throw std::out_of_range("cellmate::rebuild() extent exceeds 21-bit cell coordinates");
			}

			return static_cast<uint32_t>(coordinate);
		}

		// spread 21 coordinate bits into bit positions 0, 3, 6, ..., 60 of a 64-bit key
		static uint64_t spread_bits(uint32_t value)
		{
			uint64_t bits = value;

			bits = (bits | (bits << 32)) & 0x1f00000000ffffULL;
			bits = (bits | (bits << 16)) & 0x1f0000ff0000ffULL;
			bits = (bits | (bits << 8)) & 0x100f00f00f00f00fULL;
			bits = (bits | (bits << 4)) & 0x10c30c30c30c30c3ULL;
			bits = (bits | (bits << 2)) & 0x1249249249249249ULL;

			return bits;
		}

		// interleave three 21-bit coordinates into a single 64-bit Morton key
		static uint64_t morton_key(uint32_t x, uint32_t y, uint32_t z)
		{
			// x occupies bit positions: 0, 3, 6, ..., 60
			// y occupies bit positions: 1, 4, 7, ..., 61
			// z occupies bit positions: 2, 5, 8, ..., 62
			return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2);
		}
	};
}
