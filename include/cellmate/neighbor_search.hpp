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
		void rebuild(const float* x, const float* y, const float* z, std::size_t N_points)
		{
			positions_ = {};
			point_order_.clear();
			cell_keys_.clear();
			cell_starts_.clear();

			// validate input
			if (N_points == 0) { return; }
			if (!x || !y || !z)
			{
				throw std::invalid_argument("cellmate::rebuild() requires three coordinate arrays");
			}

			const std::array<const float*, 3> positions{ x, y, z };

			// place the local grid origin at the minimum (x,y,z) to avoid negative Morton keys
			origin_ = { x[0], y[0], z[0] };
			last_cell_ = {};
			for (std::size_t i = 0; i < N_points; i++)
			{
				if (!std::isfinite(x[i]) ||
					!std::isfinite(y[i]) ||
					!std::isfinite(z[i]))
				{
					throw std::invalid_argument("cellmate::rebuild() coordinates must be finite");
				}

				origin_[0] = (std::min)(origin_[0], x[i]);
				origin_[1] = (std::min)(origin_[1], y[i]);
				origin_[2] = (std::min)(origin_[2], z[i]);
			}

			// assign Morton keys to each point
			point_order_.resize(N_points); // TODO: find alternative that grows with extra margin to avoid repeated allocations?
			for (std::size_t i = 0; i < N_points; i++)
			{
				const uint32_t cx = cell_coord(x[i], 0);
				const uint32_t cy = cell_coord(y[i], 1);
				const uint32_t cz = cell_coord(z[i], 2);

				last_cell_[0] = (std::max)(last_cell_[0], cx);
				last_cell_[1] = (std::max)(last_cell_[1], cy);
				last_cell_[2] = (std::max)(last_cell_[2], cz);

				// store Morton key and point index from the original input array
				point_order_[i] = { morton_key(cx, cy, cz), i };
			}

			// sort point indices by Morton key
			// TODO: use faster sorting algo?
			std::sort(point_order_.begin(), point_order_.end(),
				[](const PointEntry& a, const PointEntry& b)
				{
					return a.key < b.key || (a.key == b.key && a.idx < b.idx);
				});

			// store each occupied cell's key and starting point offset
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

			positions_ = positions;
		}

		// TODO: XML documentation?
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
						const auto cell = std::lower_bound(cell_keys_.begin(), cell_keys_.end(), key);
						if (cell == cell_keys_.end() || *cell != key) { continue; }

						const std::size_t cell_idx = static_cast<std::size_t>(cell - cell_keys_.begin());

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

		double radius_;
		std::array<const float*, 3> positions_{};
		std::array<float, 3> origin_{};
		std::array<uint32_t, 3> last_cell_{};
		std::vector<PointEntry> point_order_;
		std::vector<uint64_t> cell_keys_;
		std::vector<std::size_t> cell_starts_;

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
