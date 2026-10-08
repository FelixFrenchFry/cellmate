#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "detail/simd.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// match backend settings so distance checks can inline into the query loops
CELLMATE_DETAIL_FP_BEGIN

namespace cellmate
{
	using std::uint32_t;
	using std::uint64_t;

	// controls whether operations use multithreading
	enum class ComputeMode
	{
		seq, // run serially on the calling thread
		par  // run in parallel, if OpenMP available and useful
	};

	// controls whether input data is copied or read directly
	enum class PointAccess
	{
		cpy, // copy and spatially order positions for contiguous access within cells
		ref  // read input positions directly, requiring arrays to stay alive and unchanged
	};

	// TODO: XML documentation?
	class NeighborSearch
	{
	public:

		explicit NeighborSearch(float radius, PointAccess point_access = PointAccess::cpy)
			: radius_(radius), point_access_(point_access)
		{
			if (!std::isfinite(radius) || radius <= 0)
			{
				throw std::invalid_argument("cellmate::NeighborSearch() radius must be finite and positive");
			}
			if (point_access != PointAccess::cpy && point_access != PointAccess::ref)
			{
				throw std::invalid_argument("cellmate::NeighborSearch() point access must be cpy or ref");
			}
		}

		// TODO: XML documentation?
		void rebuild(const float* x, const float* y, const float* z, std::size_t N_points, ComputeMode exec = ComputeMode::par)
		{
			positions_ = {};
			build_id_.reset();

			// point order and cell table keep their storage, so resize() does not zero-fill them
			// (each rebuild path must overwrite all entries after resize() or clear before appending)
			if (N_points == 0)
			{
				point_order_.clear();
				cell_keys_.clear();
				cell_starts_.clear();
				copied_ordered_positions_[0].clear();
				copied_ordered_positions_[1].clear();
				copied_ordered_positions_[2].clear();
				return;
			}
			if (!x || !y || !z)
			{
				throw std::invalid_argument("cellmate::rebuild() requires three coordinate arrays");
			}

			// select sequential or parallel version on execution mode and heuristics
#ifdef _OPENMP
			const int N_threads = select_thread_count(N_points, exec, 4096);
			if (exec == ComputeMode::par && N_threads > 1 &&
				N_points <= static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
			{
				rebuild_par(x, y, z, static_cast<std::ptrdiff_t>(N_points), N_threads);
			}
			else
			{
				rebuild_seq(x, y, z, N_points);
			}
#else
			(void)exec;
			rebuild_seq(x, y, z, N_points);
#endif

			if (point_access_ == PointAccess::cpy) { copy_positions(x, y, z, N_points, exec); }
			build_id_ = std::make_shared<QueryContext::BuildId>();
			if (point_access_ == PointAccess::ref) { positions_ = { x, y, z }; }
		}

		// TODO: XML documentation?
		void query(std::array<float, 3> query_pos, std::vector<std::size_t>& neighbors) const
		{
			neighbors.clear();

			if (!std::isfinite(query_pos[0]) || !std::isfinite(query_pos[1]) || !std::isfinite(query_pos[2]))
			{
				throw std::invalid_argument("cellmate::query() query position must be finite");
			}
			if (!build_id_) { return; }

			append_neighbors(query_pos, thread_context(), neighbors);
		}

		// TODO: XML documentation?
		void query_batch(const float* x, const float* y, const float* z, std::size_t N_queries,
			std::vector<std::size_t>& neighbors, std::vector<std::size_t>& offsets, ComputeMode exec = ComputeMode::par) const
		{
			if (&neighbors == &offsets)
			{
				throw std::invalid_argument("cellmate::query_batch() requires distinct output vectors");
			}
			if (N_queries >= offsets.max_size())
			{
				throw std::length_error("cellmate::query_batch() query count exceeds output limits");
			}
			if (N_queries == 0)
			{
				neighbors.clear();
				offsets.assign(1, 0);
				return;
			}
			if (!x || !y || !z)
			{
				throw std::invalid_argument("cellmate::query_batch() requires three coordinate arrays");
			}

			// validate query positions before starting parallel work
			// TODO: disable in release builds for performance?
			for (std::size_t i = 0; i < N_queries; i++)
			{
				if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]))
				{
					throw std::invalid_argument("cellmate::query_batch() query positions must be finite");
				}
			}

			// prepare output ranges, including empty search results
			offsets.resize(N_queries + 1);
			if (!build_id_)
			{
				neighbors.clear();
				std::fill(offsets.begin(), offsets.end(), 0);
				return;
			}

			// select sequential or parallel version on execution mode and heuristics
#ifdef _OPENMP
			const int N_threads = select_thread_count(N_queries, exec, 256);
			if (exec == ComputeMode::par && N_threads > 1 &&
				N_queries <= static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
			{
				query_batch_par(x, y, z, static_cast<std::ptrdiff_t>(N_queries), neighbors, offsets, N_threads);
			}
			else
			{
				query_batch_seq(x, y, z, N_queries, neighbors, offsets);
			}
#else
			(void)exec;
			query_batch_seq(x, y, z, N_queries, neighbors, offsets);
#endif
		}

	private:

		// reusable cell lookup cache for one querying thread
		struct QueryContext
		{
			struct CellCacheEntry
			{
				// invalid Morton key marks an unused entry
				uint64_t key = (std::numeric_limits<uint64_t>::max)();
				std::size_t idx = 0;
			};

			// track build identity to prevent stale cache reuse
			struct BuildId {};
			std::shared_ptr<const BuildId> build_id_;
			std::array<CellCacheEntry, 512> cell_cache_{};
		};

		// TODO: go without struct?
		struct PointEntry
		{
			uint64_t key;
			std::size_t idx;
		};

		double radius_;
		PointAccess point_access_;
		std::array<const float*, 3> positions_{};
		std::array<std::vector<float>, 3> copied_ordered_positions_;
		std::array<float, 3> origin_{};
		std::array<uint32_t, 3> last_cell_{};
		std::vector<PointEntry> point_order_;
		std::vector<uint64_t> cell_keys_;
		std::vector<std::size_t> cell_starts_;

		// reusable entry buffer for sequential and parallel radix sorting
		std::vector<PointEntry> point_order_scratch_;
		// per-chunk radix counts, then write offsets
		// (only used for parallel sorting, but always declared to keep class layout consistent)
		std::vector<std::array<std::size_t, 256>> radix_offsets_;
		// per-chunk cell counts, then write offsets for parallel cell-table construction
		std::vector<std::size_t> cell_chunk_offsets_;

		std::shared_ptr<const QueryContext::BuildId> build_id_;

		// reusable cell lookup cache of the calling thread, synchronized with the current build
		// (the returned reference must only be used on the calling thread)
		QueryContext& thread_context() const
		{
			// reuse a separate cache on each calling thread
			static thread_local QueryContext context;

			// clear cached cell indices when the context switches to a different build
			if (context.build_id_ != build_id_)
			{
				context.cell_cache_.fill(QueryContext::CellCacheEntry{});
				context.build_id_ = build_id_;
			}

			return context;
		}

		// append neighbors of one query position without clearing the output
		void append_neighbors(std::array<float, 3> query_pos, QueryContext& context, std::vector<std::size_t>& neighbors) const
		{
			// resolve position storage without retaining pointers into the vectors
			const bool copied = point_access_ == PointAccess::cpy;
			const std::array<const float*, 3> positions = copied
				? std::array<const float*, 3>{ copied_ordered_positions_[0].data(), copied_ordered_positions_[1].data(), copied_ordered_positions_[2].data() }
				: positions_;

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

			const detail::DistQuery dist_query(query_pos, radius_ * radius_);

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
						const std::size_t cell_idx = find_cell(key, context);
						if (cell_idx == cell_keys_.size()) { continue; }

						append_cell_neighbors(cell_idx, positions, copied, dist_query, neighbors);
					}
				}
			}
		}

		// collect original indices of points within the search radius
		// (filtering one cell, preserving output order of the original indices)
		void append_cell_neighbors(std::size_t cell_idx, const std::array<const float*, 3>& positions,
			bool copied, const detail::DistQuery& dist_query, std::vector<std::size_t>& neighbors) const
		{
			std::size_t i = cell_starts_[cell_idx];
			const std::size_t cell_end = cell_starts_[cell_idx + 1];
			if (copied) // optional SIMD mask distance checks for contiguous positions
			{
				for (; cell_end - i >= detail::DistQuery::N_lanes; i += detail::DistQuery::N_lanes)
				{
					const unsigned mask = dist_query.neighbor_mask(positions[0] + i, positions[1] + i, positions[2] + i);
					for (std::size_t lane = 0; lane < detail::DistQuery::N_lanes; lane++)
					{
						if (mask & (1u << lane)) { neighbors.push_back(point_order_[i + lane].idx); }
					}
				}
			}

			// scalar distance checks for non-contiguous positions
			// (or for the remainder after the SIMD check loop, if available)
			for (; i < cell_end; i++)
			{
				const std::size_t idx = copied ? i : point_order_[i].idx;
				if (dist_query.contains(positions[0][idx], positions[1][idx], positions[2][idx]))
				{
					neighbors.push_back(point_order_[i].idx);
				}
			}
		}

		void query_batch_seq(const float* x, const float* y, const float* z, std::size_t N_queries,
			std::vector<std::size_t>& neighbors, std::vector<std::size_t>& offsets) const
		{
			neighbors.clear();

			// fill each query's output range while preserving neighbor order
			QueryContext& context = thread_context();
			for (std::size_t i = 0; i < N_queries; i++)
			{
				offsets[i] = neighbors.size();
				append_neighbors({ x[i], y[i], z[i] }, context, neighbors);
			}
			offsets[N_queries] = neighbors.size();
		}

#ifdef _OPENMP
		void query_batch_par(const float* x, const float* y, const float* z, std::ptrdiff_t N_queries,
			std::vector<std::size_t>& neighbors, std::vector<std::size_t>& offsets, int N_threads) const
		{
			// divide queries into one contiguous chunk per thread
			const std::ptrdiff_t chunk_size = N_queries / N_threads + (N_queries % N_threads != 0);
			const std::ptrdiff_t N_chunks = N_queries / chunk_size + (N_queries % chunk_size != 0);
			std::vector<std::vector<std::size_t>> chunk_neighbors(N_chunks);
			std::exception_ptr error;

			// collect neighbors of each chunk in a separate buffer
			// (offsets temporarily store the start of each query within its chunk buffer)
			#pragma omp parallel for schedule(static) num_threads(N_threads)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::ptrdiff_t begin = chunk * chunk_size;
				const std::ptrdiff_t end = begin + (std::min)(chunk_size, N_queries - begin);
				try
				{
					QueryContext& context = thread_context();
					std::vector<std::size_t>& buffer = chunk_neighbors[chunk];
					for (std::ptrdiff_t i = begin; i < end; i++)
					{
						offsets[i] = buffer.size();
						append_neighbors({ x[i], y[i], z[i] }, context, buffer);
					}
				}
				catch (...)
				{
					// propagate alloc failures after parallel work finishes
					#pragma omp critical(cellmate_query_batch_error)
					{
						if (!error) { error = std::current_exception(); }
					}
				}
			}
			if (error) { std::rethrow_exception(error); }

			// convert chunk sizes into output positions in chunk order
			std::vector<std::size_t> chunk_offsets(N_chunks);
			std::size_t N_neighbors = 0;
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::size_t count = chunk_neighbors[chunk].size();
				if (count > neighbors.max_size() - N_neighbors)
				{
					throw std::length_error("cellmate::query_batch() neighbor count exceeds output limits");
				}
				chunk_offsets[chunk] = N_neighbors;
				N_neighbors += count;
			}
			offsets[N_queries] = N_neighbors;
			neighbors.resize(N_neighbors);
			if (N_neighbors == 0) { return; }

			// fill each chunk's output range and query offsets while preserving neighbor order
			#pragma omp parallel for schedule(static) num_threads(N_threads)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::ptrdiff_t begin = chunk * chunk_size;
				const std::ptrdiff_t end = begin + (std::min)(chunk_size, N_queries - begin);
				for (std::ptrdiff_t i = begin; i < end; i++)
				{
					offsets[i] += chunk_offsets[chunk];
				}
				std::copy(chunk_neighbors[chunk].begin(), chunk_neighbors[chunk].end(), neighbors.begin() + chunk_offsets[chunk]);
			}
		}
#endif

		// copy positions into cell order, while preserving the the mapping to original indices
		void copy_positions(const float* x, const float* y, const float* z, std::size_t N_points, ComputeMode exec)
		{
			copied_ordered_positions_[0].resize(N_points);
			copied_ordered_positions_[1].resize(N_points);
			copied_ordered_positions_[2].resize(N_points);

#ifdef _OPENMP
			const int N_threads = select_thread_count(N_points, exec, 4096);
			if (N_threads > 1 && N_points <= static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
			{
				#pragma omp parallel for schedule(static) num_threads(N_threads)
				for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N_points); i++)
				{
					const std::size_t idx = point_order_[i].idx;
					copied_ordered_positions_[0][i] = x[idx];
					copied_ordered_positions_[1][i] = y[idx];
					copied_ordered_positions_[2][i] = z[idx];
				}
				return;
			}
#else
			(void)exec;
#endif

			for (std::size_t i = 0; i < N_points; i++)
			{
				const std::size_t idx = point_order_[i].idx;
				copied_ordered_positions_[0][i] = x[idx];
				copied_ordered_positions_[1][i] = y[idx];
				copied_ordered_positions_[2][i] = z[idx];
			}
		}

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

			sort_point_order_seq();
			build_cell_table_seq();
		}

#ifdef _OPENMP
		// choose how many threads to use for parallel work, based on heuristics
		static int select_thread_count(std::size_t N_items, ComputeMode exec, int min_items_per_thread)
		{
			assert(min_items_per_thread > 0);
			if (exec != ComputeMode::par || omp_in_parallel()) { return 1; }

			const int max_threads = omp_get_max_threads();
			const int N_threads = static_cast<int>(
				(std::min)(N_items / min_items_per_thread, static_cast<std::size_t>(max_threads)));

			return (std::max)(1, N_threads);
		}
#endif

#ifdef _OPENMP
		void rebuild_par(const float* x, const float* y, const float* z, std::ptrdiff_t N_points, int N_threads)
		{
			// compute minimum and maximum (x,y,z)
			// (compute local bounds per thread, then merge into grid bounds)
			std::array<float, 3> minimum{ x[0], y[0], z[0] };
			std::array<float, 3> maximum = minimum;
			bool invalid = false;
			#pragma omp parallel num_threads(N_threads)
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
			#pragma omp parallel for schedule(static) num_threads(N_threads)
			for (std::ptrdiff_t i = 0; i < N_points; i++)
			{
				const uint32_t cx = cell_coord(x[i], 0);
				const uint32_t cy = cell_coord(y[i], 1);
				const uint32_t cz = cell_coord(z[i], 2);
				point_order_[i] = { morton_key(cx, cy, cz), static_cast<std::size_t>(i) };
			}

			sort_point_order_par(N_threads);
			build_cell_table_par(N_threads);
		}
#endif

#ifdef _OPENMP
		// stable radix sort of point indices by Morton key, parallel version
		void sort_point_order_par(int N_threads)
		{
			const std::size_t N_points = point_order_.size();
			const uint64_t max_key = morton_key(last_cell_[0], last_cell_[1], last_cell_[2]);
			if (N_points < 2 || max_key == 0) { return; }

			// determine how many bytes to sort
			// (1 to 8 radix passes needed for 63-bit Morton keys)
			uint32_t N_passes = 0;
			for (uint64_t key = max_key; key != 0; key >>= 8)
			{
				N_passes++;
			}

			// divide entries into chunks for parallel counting and scattering
			const std::size_t chunk_size = 4096; // TODO: find better heuristic?
			const std::ptrdiff_t N_chunks = static_cast<std::ptrdiff_t>(
				N_points / chunk_size + (N_points % chunk_size != 0));
			const int N_sort_threads = static_cast<int>(
				(std::min)(static_cast<std::ptrdiff_t>(N_threads), N_chunks));

			// prepare output buffer and per-chunk byte counts
			point_order_scratch_.resize(N_points);
			radix_offsets_.resize(N_chunks);
			std::array<std::size_t, 256> digit_offsets;
			PointEntry* src = point_order_.data();
			PointEntry* dst = point_order_scratch_.data();

			#pragma omp parallel num_threads(N_sort_threads)
			{
				// sort bytes from least to most significant
				for (uint32_t pass = 0; pass < N_passes; pass++)
				{
					const uint32_t shift = pass * 8;

					// count each byte value separately for each chunk
					#pragma omp for schedule(static)
					for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
					{
						const std::size_t begin = static_cast<std::size_t>(chunk) * chunk_size;
						const std::size_t end = begin + (std::min)(chunk_size, N_points - begin);
						std::array<std::size_t, 256>& counts = radix_offsets_[chunk];
						counts.fill(0);
						for (std::size_t i = begin; i < end; i++)
						{
							counts[(src[i].key >> shift) & 255]++;
						}
					}

					// convert chunk counts into offsets for output range of each byte
					#pragma omp for schedule(static)
					for (int digit = 0; digit < 256; digit++)
					{
						std::size_t offset = 0;
						for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
						{
							const std::size_t count = radix_offsets_[chunk][digit];
							radix_offsets_[chunk][digit] = offset;
							offset += count;
						}
						digit_offsets[digit] = offset;
					}

					// convert byte totals into starting output positions
					#pragma omp single
					{
						std::size_t offset = 0;
						for (std::size_t digit = 0; digit < 256; digit++)
						{
							const std::size_t count = digit_offsets[digit];
							digit_offsets[digit] = offset;
							offset += count;
						}
					}

					// scatter entries while preserving equal-byte order
					#pragma omp for schedule(static)
					for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
					{
						const std::size_t begin = static_cast<std::size_t>(chunk) * chunk_size;
						const std::size_t end = begin + (std::min)(chunk_size, N_points - begin);
						std::array<std::size_t, 256>& offsets = radix_offsets_[chunk];
						for (std::size_t i = begin; i < end; i++)
						{
							const std::size_t digit = (src[i].key >> shift) & 255;
							dst[digit_offsets[digit] + offsets[digit]++] = src[i];
						}
					}

					#pragma omp single
					{
						std::swap(src, dst);
					}
				}
			}

			// odd pass counts leave the sorted entries in scratch
			if (src != point_order_.data()) { point_order_.swap(point_order_scratch_); }
		}
#endif

#ifdef _OPENMP
		// fill cell table with occupied cell keys and start offsets, parallel version
		void build_cell_table_par(int N_threads)
		{
			const std::size_t N_points = point_order_.size();

			// divide entries into chunks for parallel counting and filling
			const std::size_t chunk_size = 4096;
			const std::ptrdiff_t N_chunks = static_cast<std::ptrdiff_t>(
				N_points / chunk_size + (N_points % chunk_size != 0));
			const int N_build_threads = static_cast<int>(
				(std::min)(static_cast<std::ptrdiff_t>(N_threads), N_chunks));
			cell_chunk_offsets_.resize(N_chunks);

			// count cell starts separately for each chunk
			#pragma omp parallel for schedule(static) num_threads(N_build_threads)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::size_t begin = static_cast<std::size_t>(chunk) * chunk_size;
				const std::size_t end = begin + (std::min)(chunk_size, N_points - begin);
				std::size_t count = 0;
				for (std::size_t i = begin; i < end; i++)
				{
					if (i == 0 || point_order_[i].key != point_order_[i - 1].key) { count++; }
				}
				cell_chunk_offsets_[chunk] = count;
			}

			// convert counts into output positions in chunk order
			std::size_t N_cells = 0;
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::size_t count = cell_chunk_offsets_[chunk];
				cell_chunk_offsets_[chunk] = N_cells;
				N_cells += count;
			}

			// prepare output arrays, including the final cell range end
			// (no clearing needed, every entry is overwritten below)
			cell_keys_.resize(N_cells);
			cell_starts_.resize(N_cells + 1);

			// fill each chunk's output range while preserving cell order
			#pragma omp parallel for schedule(static) num_threads(N_build_threads)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::size_t begin = static_cast<std::size_t>(chunk) * chunk_size;
				const std::size_t end = begin + (std::min)(chunk_size, N_points - begin);
				std::size_t offset = cell_chunk_offsets_[chunk];
				for (std::size_t i = begin; i < end; i++)
				{
					if (i != 0 && point_order_[i].key == point_order_[i - 1].key) { continue; }
					cell_keys_[offset] = point_order_[i].key;
					cell_starts_[offset] = i;
					offset++;
				}
			}

			cell_starts_[N_cells] = N_points; // end of the last cell
		}
#endif

		// stable radix sort of point indices by Morton key, sequential version
		void sort_point_order_seq()
		{
			const std::size_t N_points = point_order_.size();
			const uint64_t max_key = morton_key(last_cell_[0], last_cell_[1], last_cell_[2]);
			if (N_points < 2 || max_key == 0) { return; }

			// determine how many bytes to sort
			// (1 to 8 radix passes needed for 63-bit Morton keys)
			uint32_t N_passes = 0;
			for (uint64_t key = max_key; key != 0; key >>= 8)
			{
				N_passes++;
			}

			// prepare output buffer and byte counts
			point_order_scratch_.resize(N_points);
			std::array<std::size_t, 256> counts;
			PointEntry* src = point_order_.data();
			PointEntry* dst = point_order_scratch_.data();

			// sort bytes from least to most significant
			for (uint32_t pass = 0; pass < N_passes; pass++)
			{
				const uint32_t shift = pass * 8;

				// count each byte value
				counts.fill(0);
				for (std::size_t i = 0; i < N_points; i++)
				{
					counts[(src[i].key >> shift) & 255]++;
				}

				// convert counts into output positions in byte order
				std::array<std::size_t, 256>& offsets = counts;
				std::size_t offset = 0;
				for (std::size_t digit = 0; digit < 256; digit++)
				{
					const std::size_t count = counts[digit];
					offsets[digit] = offset;
					offset += count;
				}

				// scatter entries while preserving equal-byte order
				for (std::size_t i = 0; i < N_points; i++)
				{
					const std::size_t digit = (src[i].key >> shift) & 255;
					dst[offsets[digit]++] = src[i];
				}

				std::swap(src, dst);
			}

			// odd pass counts leave the sorted entries in scratch
			if (src != point_order_.data()) { point_order_.swap(point_order_scratch_); }
		}

		// fill cell table with occupied cell keys and start offsets, sequential version
		void build_cell_table_seq()
		{
			cell_keys_.clear();
			cell_starts_.clear();

			const std::size_t N_points = point_order_.size();

			// fill output range while preserving cell order
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
		std::size_t find_cell(uint64_t key, QueryContext& context) const
		{
			const std::size_t slot = static_cast<std::size_t>(key ^ (key >> 32)) & (context.cell_cache_.size() - 1);
			QueryContext::CellCacheEntry& entry = context.cell_cache_[slot];
			if (entry.key == key) { return entry.idx; }

			const std::vector<uint64_t>::const_iterator cell = std::lower_bound(cell_keys_.begin(), cell_keys_.end(), key);
			entry.key = key;
			entry.idx = cell == cell_keys_.end() || *cell != key
				? cell_keys_.size()
				: static_cast<std::size_t>(cell - cell_keys_.begin());
			
			return entry.idx;
		}

		// convert one position coordinate to a cell coordinate relative to the grid origin
		uint32_t cell_coord(float coordinate, std::size_t axis) const
		{
			const double cell_coordinate = std::floor((double(coordinate) - origin_[axis]) / radius_);

			if (cell_coordinate >= (1u << 21)) // TODO: disable in release builds?
			{
				throw std::out_of_range("cellmate::rebuild() extent exceeds 21-bit cell coordinates");
			}

			return static_cast<uint32_t>(cell_coordinate);
		}

		// spread 21 coordinate bits into bit positions 0, 3, 6, ..., 60 of a 63-bit key
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

		// interleave three 21-bit coordinates into a single 63-bit Morton key
		static uint64_t morton_key(uint32_t x, uint32_t y, uint32_t z)
		{
			// x occupies bit positions: 0, 3, 6, ..., 60
			// y occupies bit positions: 1, 4, 7, ..., 61
			// z occupies bit positions: 2, 5, 8, ..., 62
			return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2);
		}
	};
}

CELLMATE_DETAIL_FP_END
