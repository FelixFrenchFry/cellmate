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

#include "detail/index_compaction.hpp"

#if (defined(_M_X64) || defined(__x86_64__)) && !defined(_M_ARM64EC)
#include <emmintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// match backend settings so distance checks can inline into the query loops
CELLMATE_DETAIL_FP_BEGIN

namespace cellmate
{
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

	// controls the processing order of batch queries
	enum class QueryOrder
	{
		morton, // spatially reorder queries by Morton code before querying
		input   // process queries in the supplied input order
	};

	// TODO: XML documentation?
	class NeighborSearch
	{
	public:

		explicit NeighborSearch(float radius, PointAccess point_access = PointAccess::cpy)
			: radius_(radius), point_access_(point_access)
		{
			radius_squ_ = radius_ * radius_;

			if (!std::isfinite(radius) || radius <= 0)
			{
				throw std::invalid_argument("'cellmate::NeighborSearch()' 'radius' must be finite and positive");
			}
			if (!std::isnormal(radius_squ_))
			{
				throw std::invalid_argument("'cellmate::NeighborSearch()' squared 'radius' must be a finite positive normal float");
			}
			if (point_access != PointAccess::cpy && point_access != PointAccess::ref)
			{
				throw std::invalid_argument("'cellmate::NeighborSearch()' 'point_access' must be 'cpy' or 'ref'");
			}

			cell_width_inv_ = 1.0 / (double(radius_) * (1.0 + std::ldexp(1.0, -20))); // slightly wider than the radius to absorb rounding
		}

		// TODO: XML documentation?
		void rebuild(const float* x, const float* y, const float* z, std::size_t N_points, ComputeMode exec = ComputeMode::par)
		{
			positions_ = {};
			build_id_.reset();
			cell_prefix_.clear();
			cell_prefix_shift_ = 64;

			// point order and cell table keep their storage, so 'resize()' does not zero-fill them
			// (each rebuild path must overwrite all entries after 'resize()' or clear before appending)
			if (N_points == 0)
			{
				point_order_.clear();
				cell_keys_.clear();
				cell_starts_.clear();
				copied_ordered_positions_[0].clear();
				copied_ordered_positions_[1].clear();
				copied_ordered_positions_[2].clear();
				ordered_indices_.clear();
				return;
			}
			if (N_points > (std::numeric_limits<std::uint32_t>::max)())
			{
				throw std::length_error("'cellmate::rebuild()' 'N_points' exceeds 32-bit point count limits");
			}
			if (!x || !y || !z)
			{
				throw std::invalid_argument("'cellmate::rebuild()' requires three coordinate arrays");
			}

#ifdef _OPENMP
			// select sequential or parallel version on execution mode and heuristics
			const int N_threads = select_thread_count(N_points, exec, N_rebuild_points_min_par_);
			if (exec == ComputeMode::par && N_threads > 1 &&
				N_points <= static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
			{
				rebuild_par(x, y, z, N_points, N_threads);
			}
			else
			{
				rebuild_seq(x, y, z, N_points);
			}
#else
			(void)exec;
			rebuild_seq(x, y, z, N_points);
#endif

			build_cell_prefix_directory();

			if (point_access_ == PointAccess::cpy) { copy_positions(x, y, z, N_points, exec); }
			build_id_ = std::make_shared<QueryContext::BuildId>();
			if (point_access_ == PointAccess::ref) { positions_ = { x, y, z }; }
		}

		// TODO: XML documentation?
		void query(std::array<float, 3> query_pos, std::vector<std::uint32_t>& neighbors) const
		{
			neighbors.clear();

			if (!std::isfinite(query_pos[0]) || !std::isfinite(query_pos[1]) || !std::isfinite(query_pos[2]))
			{
				throw std::invalid_argument("'cellmate::query()' 'query_pos' must be finite");
			}
			if (!build_id_) { return; }

			append_neighbors(query_pos, thread_context(), neighbors);
		}

		// TODO: XML documentation?
		void query_batch(const float* x, const float* y, const float* z, std::size_t N_queries,
			std::vector<std::uint32_t>& neighbors, std::vector<std::size_t>& offsets,
			ComputeMode exec = ComputeMode::par, QueryOrder order = QueryOrder::morton) const
		{
			if (order != QueryOrder::morton && order != QueryOrder::input)
			{
				throw std::invalid_argument("'cellmate::query_batch()' 'order' must be 'morton' or 'input'");
			}
			if (N_queries >= offsets.max_size())
			{
				throw std::length_error("'cellmate::query_batch()' 'N_queries' exceeds output limits");
			}
			if (N_queries > (std::numeric_limits<std::uint32_t>::max)())
			{
				throw std::length_error("'cellmate::query_batch()' 'N_queries' exceeds 32-bit query count limits");
			}
			if (N_queries == 0)
			{
				neighbors.clear();
				offsets.assign(1, 0);
				return;
			}
			if (!x || !y || !z)
			{
				throw std::invalid_argument("'cellmate::query_batch()' requires three coordinate arrays");
			}

			// validate query positions before starting parallel work
			// TODO: disable in release builds for performance?
			for (std::size_t i = 0; i < N_queries; i++)
			{
				if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]))
				{
					throw std::invalid_argument("'cellmate::query_batch()' query positions must be finite");
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

			// retain scratch capacity across calls to avoid repeated allocations
			// (belongs to the caller, including buffers used by its parallel threads)
			// TODO: perform explicit scratch release after unusually large batches?
			static thread_local BatchScratch scratch;
			if (order == QueryOrder::morton) { scratch.query_order_.resize(N_queries); }

#ifdef _OPENMP
			// select sequential or parallel version on execution mode and heuristics
			// TODO: use proper cost metric that includes neighbor density (derived from number of points and number of occupied cells)?
			const int N_threads = select_thread_count(N_queries, exec, N_batch_queries_min_par_);
			if (exec == ComputeMode::par && N_threads > 1 &&
				N_queries <= static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
			{
				query_batch_par(x, y, z, static_cast<std::ptrdiff_t>(N_queries), scratch, neighbors, offsets, N_threads, order);
			}
			else
			{
				query_batch_seq(x, y, z, N_queries, scratch, neighbors, offsets, order);
			}
#else
			(void)exec;
			query_batch_seq(x, y, z, N_queries, scratch, neighbors, offsets, order);
#endif
		}

	private:

		// bounding box in cell coordinates
		struct CellBounds
		{
			std::array<std::uint32_t, 3> first;
			std::array<std::uint32_t, 3> last;
		};

		// reusable cell lookup cache and point ranges for one querying thread
		struct QueryContext
		{
			struct CellCacheEntry
			{
				// invalid Morton key marks an unused entry
				std::uint64_t key = (std::numeric_limits<std::uint64_t>::max)();
				std::size_t idx = 0;
			};

			// contiguous point order entries of >=1 occupied cells, excluding the end
			struct PointRange
			{
				std::uint32_t begin;
				std::uint32_t end;
			};

			// track build identity to prevent stale cache reuse
			struct BuildId {};
			std::shared_ptr<const BuildId> build_id_;
			std::array<CellCacheEntry, 512> cell_cache_{};
			CellBounds cell_bounds_{};
			std::vector<PointRange> point_ranges_;
			bool point_ranges_valid_ = false;

			// candidates staged for shared processing by multiple queries
			std::array<std::vector<float>, 3> staged_positions_;
			std::vector<std::uint32_t> staged_indices_;
			std::size_t N_staged_ = 0;
		};

		// 64-bit Morton key and original index of a point or query
		struct PointEntry
		{
			// 64-bit key split into two 32-bit parts
			// (avoids padding the 12 byte struct to 16 bytes)
			std::uint32_t key_low;
			std::uint32_t key_high;
			std::uint32_t idx;

			PointEntry() = default;
			PointEntry(std::uint64_t value, std::uint32_t index) noexcept
				: key_low(static_cast<std::uint32_t>(value)),
				  key_high(static_cast<std::uint32_t>(value >> 32)), idx(index) {}

			std::uint64_t key() const noexcept
			{
				return std::uint64_t(key_low) | (std::uint64_t(key_high) << 32);
			}
		};
		static_assert(sizeof(PointEntry) == 12, "PointEntry must occupy 12 bytes");

		// reusable batch storage, shared by successive searches on one calling thread
		// (retains capacity until thread exit, without retaining input or output pointers)
		struct BatchScratch
		{
			std::vector<PointEntry> query_order_; // Morton keys and original query indices, sorted for processing
			std::vector<PointEntry> query_order_scratch_; // alternate buffer for radix sort passes
			std::vector<std::array<std::size_t, 256>> radix_offsets_; // per-chunk byte counts and offsets for parallel radix sort
			std::vector<std::vector<std::uint32_t>> chunk_neighbors_; // neighbors collected per chunk before restoring query row order
		};

		float radius_;
		float radius_squ_;
		double cell_width_inv_; // slightly wider than the radius to absorb rounding
		PointAccess point_access_;
		std::array<const float*, 3> positions_{};
		std::array<std::vector<float>, 3> copied_ordered_positions_;
		std::vector<std::uint32_t> ordered_indices_;
		std::array<float, 3> origin_{};
		std::array<std::uint32_t, 3> last_cell_{};
		std::vector<PointEntry> point_order_;
		std::vector<std::uint64_t> cell_keys_;
		std::vector<std::uint32_t> cell_starts_;

		// reusable entry buffer for sequential and parallel radix sorting
		std::vector<PointEntry> point_order_scratch_;
		// per-chunk radix counts, then write offsets
		// (only used for parallel sorting, but always declared to keep class layout consistent)
		std::vector<std::array<std::size_t, 256>> radix_offsets_;
		// per-chunk cell counts, then write offsets for parallel cell-table construction
		std::vector<std::size_t> cell_chunk_offsets_;

		// cell prefix directory with the start index of each key range, plus sentinel
		// (restricts 'find_cell()' searches to cell keys with the same prefix bits)
		std::vector<std::size_t> cell_prefix_; // retains capacity across rebuilds
		// key shift to directory range, 64 flags directory as unused
		std::uint32_t cell_prefix_shift_ = 64;

		static constexpr std::size_t N_nbr_buffer_size_ = 64;
		static constexpr std::size_t N_queries_min_staged_ = 3;
		static constexpr std::size_t N_queries_min_scalar_staged_ = 6;
		// minimum sizes for running parallelized operations with the full team instead of sequentially
		static constexpr std::size_t N_rebuild_points_min_par_ = 8192;
		static constexpr std::size_t N_radix_sort_entries_min_par_ = 8192;
		static constexpr std::size_t N_batch_queries_min_par_ = 128;
		// bounds for entries per chunk of parallel radix sorts and cell table construction
		static constexpr std::size_t N_chunk_entries_min_ = 1024;
		static constexpr std::size_t N_chunk_entries_max_ = 4096;
		// chunks per thread and minimum queries per chunk of parallel batch searches
		static constexpr std::size_t N_batch_chunks_per_thread_ = 8;
		static constexpr std::size_t N_batch_chunk_queries_min_ = 16;

		std::shared_ptr<const QueryContext::BuildId> build_id_;

		// reusable query context of the calling thread, synchronized with the current build
		// (the returned reference must only be used on the calling thread)
		QueryContext& thread_context() const
		{
			// reuse a separate cache on each calling thread
			static thread_local QueryContext context;

			// clear cached lookups and ranges when the context switches to a different build
			if (context.build_id_ != build_id_)
			{
				context.cell_cache_.fill(QueryContext::CellCacheEntry{});
				context.point_ranges_valid_ = false;
				context.point_ranges_.clear();
				context.build_id_ = build_id_;
			}

			return context;
		}

		// append neighbors of one query position without clearing the output
		void append_neighbors(std::array<float, 3> query_pos, QueryContext& context, std::vector<std::uint32_t>& neighbors) const
		{
			CellBounds bounds;
			if (!query_cell_bounds(query_pos, bounds)) { return; }
			gather_query_ranges(bounds, context);
			if (point_access_ == PointAccess::cpy)
			{
				find_neighbors_from_ranges<true>(query_pos, context, neighbors);
			}
			else
			{
				find_neighbors_from_ranges<false>(query_pos, context, neighbors);
			}
		}

		// finds neighbors for one query using candidate ranges and SIMD distance checks
		// (appends matching point IDs to 'neighbors', preserving existing entries)
		template<bool copied>
		void find_neighbors_from_ranges(std::array<float, 3> query_pos, const QueryContext& context,
			std::vector<std::uint32_t>& neighbors) const
		{
			// select candidate coordinates
			const std::array<const float*, 3> positions = copied
				? std::array<const float*, 3>{ copied_ordered_positions_[0].data(),
											   copied_ordered_positions_[1].data(),
											   copied_ordered_positions_[2].data() }
				: positions_;

			const detail::DistQuery dist_query(query_pos, radius_squ_);
			// allow a full-width store when a candidate SIMD batch crosses the flush threshold
			std::uint32_t buffer[N_nbr_buffer_size_ + detail::DistQuery::N_lanes - 1];
			std::size_t N_buffered = 0;

			// check candidates in each range using the same query and buffer
			for (const QueryContext::PointRange& range : context.point_ranges_)
			{
				append_range_neighbors<copied>(range.begin, range.end, positions, dist_query, neighbors, buffer, N_buffered);
			}

			// flush remaining buffered IDs into the result list
			flush_neighbor_buffer(neighbors, buffer, N_buffered);
		}

		// moves buffered neighbor indices into the output
		static void flush_neighbor_buffer(
			std::vector<std::uint32_t>& neighbors,
			const std::uint32_t* buffer,
			std::size_t& N_buffered)
		{
			if (N_buffered == 0) { return; }
			neighbors.insert(neighbors.end(), buffer, buffer + N_buffered);
			N_buffered = 0;
		}

		// computes the inclusive bounding box of candidate cells for a query, clamped to the grid
		// (returns 'false' when none of the candidate cells are within the grid)
		bool query_cell_bounds(std::array<float, 3> query_pos, CellBounds& bounds) const
		{
			// determine query's cell relative to the grid origin
			// (always finite in fp64, distant queries yield empty bounds)
			const double cell_x = unclamped_cell_coord(query_pos[0], 0);
			const double cell_y = unclamped_cell_coord(query_pos[1], 1);
			const double cell_z = unclamped_cell_coord(query_pos[2], 2);

			// find candidate cell bounds, clamped to the extent of the grid
			const double first_x = (std::max)(0.0, cell_x - 1.0);
			const double first_y = (std::max)(0.0, cell_y - 1.0);
			const double first_z = (std::max)(0.0, cell_z - 1.0);
			const double last_x = (std::min)(double(last_cell_[0]), cell_x + 1.0);
			const double last_y = (std::min)(double(last_cell_[1]), cell_y + 1.0);
			const double last_z = (std::min)(double(last_cell_[2]), cell_z + 1.0);
			if (first_x > last_x || first_y > last_y || first_z > last_z) { return false; }

			bounds.first = { static_cast<std::uint32_t>(first_x),
							 static_cast<std::uint32_t>(first_y),
							 static_cast<std::uint32_t>(first_z) };
			bounds.last = { static_cast<std::uint32_t>(last_x),
							static_cast<std::uint32_t>(last_y),
							static_cast<std::uint32_t>(last_z) };
			return true;
		}

		static bool same_cell_bounds(const CellBounds& bounds, const CellBounds& other) noexcept
		{
			return bounds.first == other.first && bounds.last == other.last;
		}

		// collects occupied point ranges in traversal order, reusing identical bounds
		// TODO: instead of gathering query ranges, store cell indices explicitly and reuse subsets of it for nearby queries?
		void gather_query_ranges(const CellBounds& bounds, QueryContext& context) const
		{
			// reuse previous ranges if bounds are identical
			if (context.point_ranges_valid_ && same_cell_bounds(context.cell_bounds_, bounds))
			{
				return;
			}

			context.point_ranges_valid_ = false;
			context.point_ranges_.clear();

			// spread x and y bits once instead of per row and per cell
			assert(bounds.last[0] - bounds.first[0] < 3 && bounds.last[1] - bounds.first[1] < 3);
			std::array<std::uint64_t, 3> x_codes{};
			std::array<std::uint64_t, 3> y_codes{};
			for (std::uint32_t x = bounds.first[0]; x <= bounds.last[0]; x++)
			{
				x_codes[x - bounds.first[0]] = spread_bits(x);
			}
			for (std::uint32_t y = bounds.first[1]; y <= bounds.last[1]; y++)
			{
				y_codes[y - bounds.first[1]] = spread_bits(y) << 1;
			}

			// collect and merge point ranges from occupied cells in traversal order, skipping empty cells
			for (std::uint32_t z = bounds.first[2]; z <= bounds.last[2]; z++)
			{
				const std::uint64_t z_code = spread_bits(z) << 2;
				for (std::uint32_t y = bounds.first[1]; y <= bounds.last[1]; y++)
				{
					const std::uint64_t yz_code = z_code | y_codes[y - bounds.first[1]];
					for (std::uint32_t x = bounds.first[0]; x <= bounds.last[0]; x++)
					{
						const std::uint64_t key = yz_code | x_codes[x - bounds.first[0]];
						const std::size_t cell_idx = find_cell(key, context);
						if (cell_idx == cell_keys_.size()) { continue; }

						const std::uint32_t begin = cell_starts_[cell_idx];
						std::uint32_t end = cell_starts_[cell_idx + 1];
						// consecutive even/odd x cells have consecutive Morton keys
						if ((x & 1) == 0 && x < bounds.last[0])
						{
							x++;
							if (cell_idx + 1 < cell_keys_.size() && cell_keys_[cell_idx + 1] == key + 1)
							{
								end = cell_starts_[cell_idx + 2];
							}
						}
						// merge only forward-adjacent ranges to preserve neighbor order
						if (!context.point_ranges_.empty() && context.point_ranges_.back().end == begin)
						{
							context.point_ranges_.back().end = end;
						}
						else
						{
							context.point_ranges_.push_back({ begin, end });
						}
					}
				}
			}
			context.cell_bounds_ = bounds;
			context.point_ranges_valid_ = true;
		}

		// collects original indices of points within the search radius
		// (filtering one point range, preserving output order of the original indices)
		template<bool copied>
		void append_range_neighbors(std::uint32_t begin, std::uint32_t end, const std::array<const float*, 3>& positions,
			const detail::DistQuery& dist_query, std::vector<std::uint32_t>& neighbors,
			std::uint32_t* buffer, std::size_t& N_buffered) const
		{
			constexpr bool use_buffer = copied && detail::has_index_compaction;

			std::size_t i = begin;
			if (copied) // optional SIMD mask distance checks for contiguous positions
			{
				// check candidate points in full SIMD batches
				for (; end - i >= detail::DistQuery::N_lanes; i += detail::DistQuery::N_lanes)
				{
					unsigned mask = dist_query.neighbor_mask(positions[0] + i, positions[1] + i, positions[2] + i);
					if (use_buffer)
					{
						N_buffered += detail::pack_contiguous_indices(ordered_indices_.data() + i, mask, buffer + N_buffered);
						if (N_buffered >= N_nbr_buffer_size_)
						{
							flush_neighbor_buffer(neighbors, buffer, N_buffered);
						}
					}
					else
					{
						while (mask != 0)
						{
							const unsigned lane = detail::first_set_bit(mask);
							neighbors.push_back(ordered_indices_[i + lane]);
							mask &= mask - 1;
						}
					}
				}

				// check remaining candidate points as one masked SIMD batch
				// (masked lanes may read past the range end but never past the array end)
				if (end > i && ordered_indices_.size() - i >= detail::DistQuery::N_lanes)
				{
					unsigned mask = dist_query.neighbor_mask(positions[0] + i, positions[1] + i, positions[2] + i) & ((1u << (end - i)) - 1u);
					if (use_buffer)
					{
						N_buffered += detail::pack_contiguous_indices(ordered_indices_.data() + i, mask, buffer + N_buffered);
						if (N_buffered >= N_nbr_buffer_size_)
						{
							flush_neighbor_buffer(neighbors, buffer, N_buffered);
						}
					}
					else
					{
						while (mask != 0)
						{
							const unsigned lane = detail::first_set_bit(mask);
							neighbors.push_back(ordered_indices_[i + lane]);
							mask &= mask - 1;
						}
					}
					return;
				}
			}

			// scalar distance checks for non-contiguous positions
			// (or for remainders too close to the array end for a full SIMD batch)
			for (; i < end; i++)
			{
				const std::size_t idx = copied ? i : ordered_indices_[i];
				if (dist_query.contains(positions[0][idx], positions[1][idx], positions[2][idx]))
				{
					if (use_buffer)
					{
						buffer[N_buffered++] = ordered_indices_[i];
						if (N_buffered >= N_nbr_buffer_size_)
						{
							flush_neighbor_buffer(neighbors, buffer, N_buffered);
						}
					}
					else
					{
						neighbors.push_back(ordered_indices_[i]);
					}
				}
			}
		}

		// stages candidates for a query group using shared candidate ranges and bounding box filtering
		// (keeps candidates within the radius of the bounding box of a query group, in traversal order)
		void stage_query_group_candidates(std::array<float, 3> first, std::array<float, 3> last, QueryContext& context) const
		{
			const std::size_t N_lanes = detail::DistQuery::N_lanes;

			// prepare staging arrays for all candidates
			// (with extra room for full-width SIMD loads and stores past the end)
			std::size_t N_candidates = 0;
			for (const QueryContext::PointRange& range : context.point_ranges_)
			{
				N_candidates += range.end - range.begin;
			}
			if (context.staged_indices_.size() < N_candidates + N_lanes)
			{
				for (std::vector<float>& coords : context.staged_positions_)
				{
					coords.resize(N_candidates + N_lanes);
				}
				context.staged_indices_.resize(N_candidates + N_lanes);
			}

			float* staged[3]{ context.staged_positions_[0].data(),
							  context.staged_positions_[1].data(),
							  context.staged_positions_[2].data() };
			std::uint32_t* staged_indices = context.staged_indices_.data();
			std::size_t N_staged = 0;

			if (point_access_ == PointAccess::cpy)
			{
				// filter candidates against the box directly
				const detail::DistBox dist_box(first, last, radius_squ_);
				const float* positions[3]{ copied_ordered_positions_[0].data(),
										   copied_ordered_positions_[1].data(),
										   copied_ordered_positions_[2].data() };
				for (const QueryContext::PointRange& range : context.point_ranges_)
				{
					// check candidates in full SIMD batches, masking the range's last batch
					// (masked lanes may read past the range end, never past the array end)
					std::size_t i = range.begin;
					while (i < range.end && ordered_indices_.size() - i >= N_lanes)
					{
						unsigned mask = dist_box.neighbor_mask(positions[0] + i, positions[1] + i, positions[2] + i);
						if (range.end - i < N_lanes) { mask &= (1u << (range.end - i)) - 1u; }
						detail::pack_contiguous_floats(positions[0] + i, mask, staged[0] + N_staged);
						detail::pack_contiguous_floats(positions[1] + i, mask, staged[1] + N_staged);
						detail::pack_contiguous_floats(positions[2] + i, mask, staged[2] + N_staged);
						N_staged += detail::pack_contiguous_indices(ordered_indices_.data() + i, mask, staged_indices + N_staged);
						i += N_lanes;
					}

					// stage leftover candidates at the array end without filtering
					for (; i < range.end; i++)
					{
						staged[0][N_staged] = positions[0][i];
						staged[1][N_staged] = positions[1][i];
						staged[2][N_staged] = positions[2][i];
						staged_indices[N_staged] = ordered_indices_[i];
						N_staged++;
					}
				}
			}
			else // 'PointAccess::ref'
			{
				// filter candidates against the box while gathering them through the point order
				// (rejected candidates are overwritten by the next one)
				const detail::DistBoxScalar scalar_box(first, last, radius_squ_);
				for (const QueryContext::PointRange& range : context.point_ranges_)
				{
					for (std::size_t i = range.begin; i < range.end; i++)
					{
						const std::uint32_t idx = ordered_indices_[i];
						const float px = positions_[0][idx], py = positions_[1][idx], pz = positions_[2][idx];
						staged[0][N_staged] = px;
						staged[1][N_staged] = py;
						staged[2][N_staged] = pz;
						staged_indices[N_staged] = idx;
						N_staged += scalar_box.contains(px, py, pz);
					}
				}
			}
			context.N_staged_ = N_staged;
		}

		// finds neighbors for one query using staged candidates and the selected distance check backend
		// (appends matching point IDs to 'neighbors', preserving existing entries)
		void find_neighbors_from_staged(std::array<float, 3> query_pos, const QueryContext& context,
			std::vector<std::uint32_t>& neighbors) const
		{
			const std::size_t N_lanes = detail::DistQuery::N_lanes;

			// select staged candidates
			const float* staged[3]{ context.staged_positions_[0].data(),
									context.staged_positions_[1].data(),
									context.staged_positions_[2].data() };
			const std::uint32_t* staged_indices = context.staged_indices_.data();
			const std::size_t N_staged = context.N_staged_;

			// check candidate points in scalar way, skip SIMD batching stuff
			const detail::DistQuery dist_query(query_pos, radius_squ_);
			if (detail::is_scalar_backend)
			{
				for (std::size_t i = 0; i < N_staged; i++)
				{
					if (dist_query.contains(staged[0][i], staged[1][i], staged[2][i]))
					{
						neighbors.push_back(staged_indices[i]);
					}
				}
				return;
			}

			// allow a full-width store when a candidate SIMD batch crosses the flush threshold
			std::uint32_t buffer[N_nbr_buffer_size_ + detail::DistQuery::N_lanes - 1];
			std::size_t N_buffered = 0;

			// check candidate points in full SIMD batches
			std::size_t i = 0;
			for (; N_staged - i >= N_lanes; i += N_lanes)
			{
				const unsigned mask = dist_query.neighbor_mask(staged[0] + i, staged[1] + i, staged[2] + i);
				N_buffered += detail::pack_contiguous_indices(staged_indices + i, mask, buffer + N_buffered);
				if (N_buffered >= N_nbr_buffer_size_)
				{
					flush_neighbor_buffer(neighbors, buffer, N_buffered);
				}
			}

			// check remaining candidate points as one masked SIMD batch
			// (masked lanes read into the padding past the staged candidates)
			if (N_staged > i)
			{
				const unsigned mask = dist_query.neighbor_mask(staged[0] + i, staged[1] + i, staged[2] + i) & ((1u << (N_staged - i)) - 1u);
				N_buffered += detail::pack_contiguous_indices(staged_indices + i, mask, buffer + N_buffered);
			}

			// flush remaining buffered IDs into the result list
			flush_neighbor_buffer(neighbors, buffer, N_buffered);
		}

		// finds groups of queries in the range that share the same candidate cell bounds and processes
		// them either with shared staged candidates or individually, decision based on group size
		// (appends neighbor lists to 'buffer', storing each count in 'offsets' at its original query index)
		void query_batch_morton_range(const float* x, const float* y, const float* z,
			const std::vector<PointEntry>& query_order, std::size_t begin, std::size_t end,
			QueryContext& context, std::vector<std::uint32_t>& buffer, std::vector<std::size_t>& offsets) const
		{
			std::size_t i = begin;
			while (i < end)
			{
				// get candidate cell bounds for current query
				const PointEntry& entry = query_order[i];
				CellBounds bounds;
				if (!query_cell_bounds({ x[entry.idx], y[entry.idx], z[entry.idx] }, bounds))
				{
					offsets[entry.idx] = 0;
					i++;
					continue;
				}

				// gather candidate point ranges within the candidate cell bounds
				gather_query_ranges(bounds, context);

				// collect consecutive queries with the same candidate cell bounds into a group
				// (stops at a different key or the end of the chunk, queries outside the grid stay alone)
				std::size_t group_end = i + 1;
				if ((entry.key() & 1u) == 0)
				{
					while (group_end < end && query_order[group_end].key() == entry.key()) { group_end++; }
				}

				// use shared candidate staging when the query group can amortize its preparation cost
				const bool use_candidate_staging = group_end - i >= (detail::is_scalar_backend
					? N_queries_min_scalar_staged_
					: N_queries_min_staged_);

				// process a large group's queries together with staged candidates, or
				// process a small group's queries individually with the gathered ranges
				if (use_candidate_staging) // large group
				{
					// stage candidates within the radius of the group's bounding box
					std::array<float, 3> first{ x[entry.idx], y[entry.idx], z[entry.idx] };
					std::array<float, 3> last = first;
					for (std::size_t k = i + 1; k < group_end; k++)
					{
						const std::uint32_t idx = query_order[k].idx;
						first = { (std::min)(first[0], x[idx]),
								  (std::min)(first[1], y[idx]),
								  (std::min)(first[2], z[idx]) };
						last = { (std::max)(last[0], x[idx]),
								 (std::max)(last[1], y[idx]),
								 (std::max)(last[2], z[idx]) };
					}
					stage_query_group_candidates(first, last, context);

					for (std::size_t k = i; k < group_end; k++)
					{
						const std::uint32_t idx = query_order[k].idx;
						const std::size_t start = buffer.size();
						find_neighbors_from_staged({ x[idx], y[idx], z[idx] }, context, buffer);
						offsets[idx] = buffer.size() - start;
					}
				}
				else // small group
				{
					for (std::size_t k = i; k < group_end; k++)
					{
						const std::uint32_t idx = query_order[k].idx;
						const std::size_t start = buffer.size();
						if (point_access_ == PointAccess::cpy)
						{
							find_neighbors_from_ranges<true>({ x[idx], y[idx], z[idx] }, context, buffer);
						}
						else
						{
							find_neighbors_from_ranges<false>({ x[idx], y[idx], z[idx] }, context, buffer);
						}
						offsets[idx] = buffer.size() - start;
					}
				}
				i = group_end;
			}
		}

		// restores neighbor lists collected in Morton order to the original query order
		// (copies each query's IDs from 'buffer' into 'neighbors' at its output offset)
		static void restore_query_results_order(const std::vector<PointEntry>& query_order, std::size_t begin, std::size_t end,
			const std::vector<std::uint32_t>& buffer, const std::vector<std::size_t>& offsets,
			std::vector<std::uint32_t>& neighbors)
		{
			std::size_t source_offset = 0;
			for (std::size_t i = begin; i < end; i++)
			{
				const std::uint32_t idx = query_order[i].idx;
				const std::size_t count = offsets[idx + 1] - offsets[idx];
				if (count == 0) { continue; }
				// copy IDs to the query's output position
				std::copy_n(buffer.begin() + source_offset, count, neighbors.begin() + offsets[idx]);
				source_offset += count;
			}
		}

		// TODO: rework to avoid temporary neighbor buffer duplication (huge memory peak)?
		void query_batch_seq(const float* x, const float* y, const float* z, std::uint32_t N_queries,
			BatchScratch& scratch,
			std::vector<std::uint32_t>& neighbors,
			std::vector<std::size_t>& offsets,
			QueryOrder order) const
		{
			if (order == QueryOrder::input)
			{
				// append directly to output, without temporary storage or row restoration
				neighbors.clear();
				offsets[0] = 0;
				QueryContext& context = thread_context();
				for (std::uint32_t i = 0; i < N_queries; i++)
				{
					append_neighbors({ x[i], y[i], z[i] }, context, neighbors);
					offsets[i + 1] = neighbors.size();
				}
				return;
			}

			// assign Morton keys and original indices to each query
			std::vector<PointEntry>& query_order = scratch.query_order_;
			for (std::uint32_t i = 0; i < N_queries; i++)
			{
				query_order[i] = { query_morton_key({ x[i], y[i], z[i] }), i };
			}

			// sort query indices by Morton key
			sort_point_order_seq(query_order, scratch.query_order_scratch_,
				(morton_key(last_cell_[0], last_cell_[1], last_cell_[2]) << 1) | 1u);

			if (scratch.chunk_neighbors_.empty()) { scratch.chunk_neighbors_.resize(1); }
			std::vector<std::uint32_t>& buffer = scratch.chunk_neighbors_[0];
			buffer.clear();
			QueryContext& context = thread_context();

			query_batch_morton_range(x, y, z, query_order, 0, N_queries, context, buffer, offsets);

			// convert counts into output positions in the caller's original query order
			// (their sum is already bounded by the size of the single neighbor buffer)
			std::size_t N_neighbors = 0;
			for (std::uint32_t i = 0; i < N_queries; i++)
			{
				const std::size_t count = offsets[i];
				offsets[i] = N_neighbors;
				N_neighbors += count;
			}
			offsets[N_queries] = N_neighbors;
			neighbors.resize(N_neighbors);
			if (N_neighbors == 0) { return; }

			restore_query_results_order(query_order, 0, N_queries, buffer, offsets, neighbors);
		}

#ifdef _OPENMP
		// TODO: rework to avoid temporary neighbor buffer duplication (huge memory peak)?
		void query_batch_par(const float* x, const float* y, const float* z, std::ptrdiff_t N_queries,
			BatchScratch& scratch,
			std::vector<std::uint32_t>& neighbors,
			std::vector<std::size_t>& offsets,
			int N_threads,
			QueryOrder order) const
		{
			std::vector<PointEntry>& query_order = scratch.query_order_;
			if (order == QueryOrder::morton)
			{
				// assign Morton keys and original indices to each query in parallel
				#pragma omp parallel for schedule(static) num_threads(N_threads)
				for (std::ptrdiff_t i = 0; i < N_queries; i++)
				{
					query_order[i] = { query_morton_key({ x[i], y[i], z[i] }), static_cast<std::uint32_t>(i) };
				}

				// sort query indices by Morton key
				const std::uint64_t max_key = (morton_key(last_cell_[0], last_cell_[1], last_cell_[2]) << 1) | 1u;
				// TODO: come up with proper cost measure heuristic to decide between seq and parallel?
				if (query_order.size() >= N_radix_sort_entries_min_par_)
				{
					sort_point_order_par(query_order, scratch.query_order_scratch_, scratch.radix_offsets_, max_key, N_threads);
				}
				else
				{
					sort_point_order_seq(query_order, scratch.query_order_scratch_, max_key);
				}
			}

			// divide queries into several contiguous chunks per thread
			const std::ptrdiff_t N_chunks_target = static_cast<std::ptrdiff_t>(N_threads) * static_cast<std::ptrdiff_t>(N_batch_chunks_per_thread_);
			std::ptrdiff_t chunk_size = N_queries / N_chunks_target + (N_queries % N_chunks_target != 0);
			if (chunk_size < static_cast<std::ptrdiff_t>(N_batch_chunk_queries_min_))
			{
				chunk_size = static_cast<std::ptrdiff_t>(N_batch_chunk_queries_min_);
			}
			const std::ptrdiff_t N_chunks = N_queries / chunk_size + (N_queries % chunk_size != 0);
			std::vector<std::vector<std::uint32_t>>& chunk_neighbors = scratch.chunk_neighbors_;
			// keep inactive chunk buffers when a batch uses fewer chunks
			if (chunk_neighbors.size() < static_cast<std::size_t>(N_chunks))
			{
				chunk_neighbors.resize(N_chunks);
			}
			std::exception_ptr error;

			// collect neighbors of each chunk in a separate buffer
			// ('offsets' temporarily store each original query's neighbor count)
			#pragma omp parallel for schedule(dynamic) num_threads(N_threads) // dynamic due to uneven query costs
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::ptrdiff_t begin = chunk * chunk_size;
				const std::ptrdiff_t end = begin + (std::min)(chunk_size, N_queries - begin);
				try
				{
					QueryContext& context = thread_context();
					std::vector<std::uint32_t>& buffer = chunk_neighbors[chunk];
					buffer.clear();
					if (order == QueryOrder::morton)
					{
						query_batch_morton_range(x, y, z, query_order, static_cast<std::size_t>(begin),
							static_cast<std::size_t>(end), context, buffer, offsets);
					}
					else
					{
						for (std::ptrdiff_t i = begin; i < end; i++)
						{
							const std::size_t idx = static_cast<std::size_t>(i);
							const std::size_t start = buffer.size();
							append_neighbors({ x[idx], y[idx], z[idx] }, context, buffer);
							offsets[idx] = buffer.size() - start;
						}
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

			// convert counts into output positions in the caller's original query order
			std::size_t N_neighbors = 0;
			for (std::ptrdiff_t i = 0; i < N_queries; i++)
			{
				const std::size_t count = offsets[i];
				if (count > neighbors.max_size() - N_neighbors)
				{
					throw std::length_error("'cellmate::query_batch()' neighbor count exceeds output limits");
				}
				offsets[i] = N_neighbors;
				N_neighbors += count;
			}
			offsets[N_queries] = N_neighbors;
			neighbors.resize(N_neighbors);
			if (N_neighbors == 0) { return; }

			// restore query rows to their original positions while preserving neighbor order
			#pragma omp parallel for schedule(static) num_threads(N_threads)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::ptrdiff_t begin = chunk * chunk_size;
				const std::ptrdiff_t end = begin + (std::min)(chunk_size, N_queries - begin);
				const std::vector<std::uint32_t>& buffer = chunk_neighbors[chunk];
				if (order == QueryOrder::morton)
				{
					restore_query_results_order(query_order, static_cast<std::size_t>(begin), static_cast<std::size_t>(end),
						buffer, offsets, neighbors);
				}
				else
				{
					// each chunk already contains consecutive output rows
					std::copy(buffer.begin(), buffer.end(), neighbors.begin() + offsets[begin]);
				}
			}
		}
#endif

		// copy positions into cell order, matching the ordered original indices
		void copy_positions(const float* x, const float* y, const float* z, std::uint32_t N_points, ComputeMode exec)
		{
			copied_ordered_positions_[0].resize(N_points);
			copied_ordered_positions_[1].resize(N_points);
			copied_ordered_positions_[2].resize(N_points);

#ifdef _OPENMP
			const int N_threads = select_thread_count(N_points, exec, N_rebuild_points_min_par_);
			if (N_threads > 1 && N_points <= static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()))
			{
				#pragma omp parallel for schedule(static) num_threads(N_threads)
				for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N_points); i++)
				{
					const std::uint32_t idx = ordered_indices_[i];
					copied_ordered_positions_[0][i] = x[idx];
					copied_ordered_positions_[1][i] = y[idx];
					copied_ordered_positions_[2][i] = z[idx];
				}
				return;
			}
#else
			(void)exec;
#endif
			for (std::uint32_t i = 0; i < N_points; i++)
			{
				const std::uint32_t idx = ordered_indices_[i];
				copied_ordered_positions_[0][i] = x[idx];
				copied_ordered_positions_[1][i] = y[idx];
				copied_ordered_positions_[2][i] = z[idx];
			}
		}

		void rebuild_seq(const float* x, const float* y, const float* z, std::uint32_t N_points)
		{
			// compute minimum and maximum (x,y,z)
			std::array<float, 3> minimum{ x[0], y[0], z[0] };
			std::array<float, 3> maximum = minimum;
			for (std::uint32_t i = 0; i < N_points; i++)
			{
				if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]))
				{
					throw std::invalid_argument("'cellmate::rebuild()' coordinates must be finite");
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
			for (std::uint32_t i = 0; i < N_points; i++)
			{
				const std::uint32_t cx = cell_coord(x[i], 0);
				const std::uint32_t cy = cell_coord(y[i], 1);
				const std::uint32_t cz = cell_coord(z[i], 2);
				point_order_[i] = { morton_key(cx, cy, cz), i };
			}

			sort_point_order_seq(point_order_, point_order_scratch_,
				morton_key(last_cell_[0], last_cell_[1], last_cell_[2]));
			build_cell_table_seq(N_points);
		}

#ifdef _OPENMP
		// chooses between sequential or parallel execution with full thread count, based on a minimum item count heuristic
		// (never selects a smaller team, since changing team sizes between parallel regions is expensive)
		static int select_thread_count(std::size_t N_items, ComputeMode exec, std::size_t N_items_min_par)
		{
			assert(N_items_min_par > 0);
			if (exec != ComputeMode::par || omp_in_parallel() || N_items < N_items_min_par) { return 1; }

			return omp_get_max_threads();
		}

		// chooses the entries per chunk, aiming for one chunk per thread, with chunks of equal size
		// (bounds limit per-chunk overhead for small chunks, and keep large inputs at the largest chunk size,
		// while equalizing chunk sizes may undercut the lower bound slightly)
		static std::size_t select_chunk_size(std::size_t N_entries, int N_threads)
		{
			assert(N_threads > 0);
			if (N_entries == 0) { return N_chunk_entries_min_; }

			// one chunk per thread, within bounds
			const std::size_t N_chunks_target = static_cast<std::size_t>(N_threads);
			std::size_t chunk_size = N_entries / N_chunks_target + (N_entries % N_chunks_target != 0);
			if (chunk_size < N_chunk_entries_min_) { chunk_size = N_chunk_entries_min_; }
			if (chunk_size > N_chunk_entries_max_) { chunk_size = N_chunk_entries_max_; }

			// equalize chunk sizes
			const std::size_t N_chunks = N_entries / chunk_size + (N_entries % chunk_size != 0);
			return N_entries / N_chunks + (N_entries % N_chunks != 0);
		}
#endif

#ifdef _OPENMP
		void rebuild_par(const float* x, const float* y, const float* z, std::uint32_t N_points, int N_threads)
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
				for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N_points); i++)
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
				throw std::invalid_argument("'cellmate::rebuild()' coordinates must be finite");
			}

			// set grid origin at the minimum (x,y,z) to avoid negative Morton keys and validate grid extent
			origin_ = minimum;
			last_cell_ = { cell_coord(maximum[0], 0), cell_coord(maximum[1], 1), cell_coord(maximum[2], 2) };

			// assign Morton keys and original indices to each point in parallel
			point_order_.resize(N_points);
			#pragma omp parallel for schedule(static) num_threads(N_threads)
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N_points); i++)
			{
				const std::uint32_t cx = cell_coord(x[i], 0);
				const std::uint32_t cy = cell_coord(y[i], 1);
				const std::uint32_t cz = cell_coord(z[i], 2);
				point_order_[i] = { morton_key(cx, cy, cz), static_cast<std::uint32_t>(i) };
			}

			if (point_order_.size() >= N_radix_sort_entries_min_par_)
			{
				sort_point_order_par(point_order_, point_order_scratch_, radix_offsets_,
					morton_key(last_cell_[0], last_cell_[1], last_cell_[2]), N_threads);
			}
			else
			{
				sort_point_order_seq(point_order_, point_order_scratch_,
					morton_key(last_cell_[0], last_cell_[1], last_cell_[2]));
			}
			build_cell_table_par(N_points, N_threads);
		}
#endif

#ifdef _OPENMP
		// stable radix sort of point indices by Morton key, parallel version
		static void sort_point_order_par(
			std::vector<PointEntry>& point_order,
			std::vector<PointEntry>& point_order_scratch,
			std::vector<std::array<std::size_t, 256>>& radix_offsets,
			std::uint64_t max_key, int N_threads)
		{
			const std::size_t N_points = point_order.size();
			if (N_points < 2 || max_key == 0) { return; }

			// determine how many bytes to sort
			// (1 to 8 radix passes needed for point and query keys)
			std::uint32_t N_passes = 0;
			for (std::uint64_t key = max_key; key != 0; key >>= 8)
			{
				N_passes++;
			}

			// divide entries into chunks for parallel counting and scattering
			const std::size_t chunk_size = select_chunk_size(N_points, N_threads);
			const std::ptrdiff_t N_chunks = static_cast<std::ptrdiff_t>(
				N_points / chunk_size + (N_points % chunk_size != 0));

			// prepare output buffer and per-chunk byte counts
			point_order_scratch.resize(N_points);
			radix_offsets.resize(N_chunks);
			std::array<std::size_t, 256> digit_offsets;
			PointEntry* src = point_order.data();
			PointEntry* dst = point_order_scratch.data();

			// parallel region for sorting pipeline
			#pragma omp parallel num_threads(N_threads)
			{
				// sort bytes from least to most significant
				for (std::uint32_t pass = 0; pass < N_passes; pass++)
				{
					const std::uint32_t shift = pass * 8;

					// count each byte value separately for each chunk
					#pragma omp for schedule(static)
					for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
					{
						const std::size_t begin = static_cast<std::size_t>(chunk) * chunk_size;
						const std::size_t end = begin + (std::min)(chunk_size, N_points - begin);
						std::array<std::size_t, 256>& counts = radix_offsets[chunk];
						counts.fill(0);
						for (std::size_t i = begin; i < end; i++)
						{
							counts[(src[i].key() >> shift) & 255]++;
						}
					}

					// convert chunk counts into offsets for output range of each byte
					#pragma omp for schedule(static)
					for (int digit = 0; digit < 256; digit++)
					{
						std::size_t offset = 0;
						for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
						{
							const std::size_t count = radix_offsets[chunk][digit];
							radix_offsets[chunk][digit] = offset;
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
						std::array<std::size_t, 256>& offsets = radix_offsets[chunk];
						for (std::size_t i = begin; i < end; i++)
						{
							const std::size_t digit = (src[i].key() >> shift) & 255;
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
			if (src != point_order.data()) { point_order.swap(point_order_scratch); }
		}
#endif

#ifdef _OPENMP
		// fill cell table with occupied cell keys, start offsets, and ordered original indices, parallel version
		void build_cell_table_par(std::uint32_t N_points, int N_threads)
		{
			// divide entries into chunks for parallel counting and filling
			// (chunk sizes never exceed the 32-bit point count)
			const std::uint32_t chunk_size = static_cast<std::uint32_t>(select_chunk_size(N_points, N_threads));
			const std::ptrdiff_t N_chunks = static_cast<std::ptrdiff_t>(
				N_points / chunk_size + (N_points % chunk_size != 0));
			cell_chunk_offsets_.resize(N_chunks);

			// count cell starts separately for each chunk
			#pragma omp parallel for schedule(static) num_threads(N_threads)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::uint32_t begin = static_cast<std::uint32_t>(chunk) * chunk_size;
				const std::uint32_t end = begin + (std::min)(chunk_size, N_points - begin);
				std::size_t count = 0;
				for (std::uint32_t i = begin; i < end; i++)
				{
					if (i == 0 || point_order_[i].key() != point_order_[i - 1].key()) { count++; }
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
			ordered_indices_.resize(N_points);

			// fill each chunk's output range while preserving cell order
			#pragma omp parallel for schedule(static) num_threads(N_threads)
			for (std::ptrdiff_t chunk = 0; chunk < N_chunks; chunk++)
			{
				const std::uint32_t begin = static_cast<std::uint32_t>(chunk) * chunk_size;
				const std::uint32_t end = begin + (std::min)(chunk_size, N_points - begin);
				std::size_t offset = cell_chunk_offsets_[chunk];
				for (std::uint32_t i = begin; i < end; i++)
				{
					ordered_indices_[i] = point_order_[i].idx;
					if (i != 0 && point_order_[i].key() == point_order_[i - 1].key()) { continue; }
					cell_keys_[offset] = point_order_[i].key();
					cell_starts_[offset] = i;
					offset++;
				}
			}

			cell_starts_[N_cells] = N_points; // end of the last cell
		}
#endif

		// stable radix sort of point indices by Morton key, sequential version
		static void sort_point_order_seq(
			std::vector<PointEntry>& point_order,
			std::vector<PointEntry>& point_order_scratch,
			std::uint64_t max_key)
		{
			const std::size_t N_points = point_order.size();
			if (N_points < 2 || max_key == 0) { return; }

			// determine how many bytes to sort
			// (1 to 8 radix passes needed for point and query keys)
			std::uint32_t N_passes = 0;
			for (std::uint64_t key = max_key; key != 0; key >>= 8)
			{
				N_passes++;
			}

			// prepare output buffer and byte counts
			point_order_scratch.resize(N_points);
			std::array<std::size_t, 256> counts;
			PointEntry* src = point_order.data();
			PointEntry* dst = point_order_scratch.data();

			// sort bytes from least to most significant
			for (std::uint32_t pass = 0; pass < N_passes; pass++)
			{
				const std::uint32_t shift = pass * 8;

				// count each byte value
				counts.fill(0);
				for (std::size_t i = 0; i < N_points; i++)
				{
					counts[(src[i].key() >> shift) & 255]++;
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
					const std::size_t digit = (src[i].key() >> shift) & 255;
					dst[offsets[digit]++] = src[i];
				}

				std::swap(src, dst);
			}

			// odd pass counts leave the sorted entries in scratch
			if (src != point_order.data()) { point_order.swap(point_order_scratch); }
		}

		// fill cell table with occupied cell keys, start offsets, and ordered original indices, sequential version
		void build_cell_table_seq(std::uint32_t N_points)
		{
			cell_keys_.clear();
			cell_starts_.clear();
			ordered_indices_.resize(N_points);

			// fill output range while preserving cell order
			// (closing the last cell after the loop, keeping 'end' within 32 bits)
			std::uint32_t begin = 0;
			ordered_indices_[0] = point_order_[0].idx;
			for (std::uint32_t end = 1; end < N_points; end++)
			{
				ordered_indices_[end] = point_order_[end].idx;
				if (point_order_[end].key() != point_order_[begin].key())
				{
					cell_keys_.push_back(point_order_[begin].key());
					cell_starts_.push_back(begin);
					begin = end;
				}
			}
			cell_keys_.push_back(point_order_[begin].key());
			cell_starts_.push_back(begin);

			cell_starts_.push_back(N_points); // end of the last cell
		}

		void build_cell_prefix_directory()
		{
			const std::size_t N_cells = cell_keys_.size();
			if (N_cells < 2048) { return; }

			// choose directory size based on occupied cell count heuristics
			constexpr std::size_t N_prefixes_min = 512;
			constexpr std::size_t N_prefixes_max = 131072;
			constexpr std::size_t N_prefix_growth = 2;
			constexpr std::size_t N_cells_per_prefix = 4;

			const std::size_t N_prefixes_target = N_cells / N_cells_per_prefix + (N_cells % N_cells_per_prefix != 0);
			std::size_t N_prefixes = N_prefixes_min;
			while (N_prefixes < N_prefixes_max && N_prefixes < N_prefixes_target)
			{
				N_prefixes = (std::min)(N_prefixes * N_prefix_growth, N_prefixes_max);
			}
			cell_prefix_.resize(N_prefixes + 1);

			std::uint32_t shift = 0;
			for (std::uint64_t remaining = cell_keys_.back() / N_prefixes; remaining != 0; remaining >>= 1)
			{
				shift++;
			}

			auto first = cell_keys_.begin();
			for (std::size_t prefix = 0; prefix < N_prefixes; prefix++)
			{
				const std::uint64_t threshold = static_cast<std::uint64_t>(prefix) << shift;
				first = std::lower_bound(first, cell_keys_.end(), threshold);
				cell_prefix_[prefix] = static_cast<std::size_t>(first - cell_keys_.begin());
			}
			cell_prefix_[N_prefixes] = cell_keys_.size();
			cell_prefix_shift_ = shift;
		}

		// reuse the cached result if its key matches, otherwise search and replace it
		std::size_t find_cell(std::uint64_t key, QueryContext& context) const
		{
			const std::size_t slot = static_cast<std::size_t>(key ^ (key >> 32)) & (context.cell_cache_.size() - 1);
			QueryContext::CellCacheEntry& entry = context.cell_cache_[slot];
			if (entry.key == key) { return entry.idx; }

			const std::uint64_t* first = cell_keys_.data();
			const std::uint64_t* cell = first;
			const std::uint64_t* last = first + cell_keys_.size();

			// restrict the search to the selected prefix bucket
			if (cell_prefix_shift_ < 64)
			{
				const std::uint64_t prefix = key >> cell_prefix_shift_;
				if (prefix >= cell_prefix_.size() - 1)
				{
					entry.key = key;
					entry.idx = cell_keys_.size();
					return entry.idx;
				}

				cell += cell_prefix_[static_cast<std::size_t>(prefix)];
				last = first + cell_prefix_[static_cast<std::size_t>(prefix) + 1];
			}

			// compare small ranges directly and cache the matching cell or miss
			std::size_t count = static_cast<std::size_t>(last - cell);
			constexpr std::size_t N_linear = 2;
			if (count <= N_linear)
			{
				unsigned mask = 0;
				for (std::size_t i = 0; i < count; i++)
				{
					mask |= static_cast<unsigned>(cell[i] == key) << i;
				}
				entry.key = key;
				entry.idx = mask == 0
					? cell_keys_.size()
					: static_cast<std::size_t>(cell - first) + detail::first_set_bit(mask);
				return entry.idx;
			}

			// shrink the range independently of the key comparison
			// (allows conditional moves instead of unpredictable branches)
			while (count > 1)
			{
				const std::size_t half = count / 2;
				cell = cell[half] < key ? cell + half : cell;
				count -= half;
			}
			cell += *cell < key;
			entry.key = key;
			entry.idx = cell == last || *cell != key
				? cell_keys_.size()
				: static_cast<std::size_t>(cell - first);
			
			return entry.idx;
		}

		// converts one position coordinate to an unclamped fp64 cell coordinate
		double unclamped_cell_coord(float coordinate, std::size_t axis) const
		{
			return std::floor((double(coordinate) - double(origin_[axis])) * cell_width_inv_);
		}

		// converts one position coordinate to a cell coordinate relative to the grid origin
		std::uint32_t cell_coord(float coordinate, std::size_t axis) const
		{
			const double cell_coordinate = unclamped_cell_coord(coordinate, axis);

			if (cell_coordinate >= (1u << 21)) // TODO: disable in release builds?
			{
				throw std::out_of_range("'cellmate::rebuild()' extent exceeds 21-bit cell coordinates");
			}

			return static_cast<std::uint32_t>(cell_coordinate);
		}

		// computes the Morton sorting key for a query, clamped to the grid
		// (uses the low bit to mark if query's cell is outside the grid: 0 = inside, 1 = outside)
		std::uint64_t query_morton_key(std::array<float, 3> query_pos) const
		{
			// determine query's cell relative to the grid origin
			const double cell_x = unclamped_cell_coord(query_pos[0], 0);
			const double cell_y = unclamped_cell_coord(query_pos[1], 1);
			const double cell_z = unclamped_cell_coord(query_pos[2], 2);

			// check whether query's cell is outside the grid before clamping
			const bool outside_grid = cell_x < 0.0 || cell_x > double(last_cell_[0]) ||
									  cell_y < 0.0 || cell_y > double(last_cell_[1]) ||
									  cell_z < 0.0 || cell_z > double(last_cell_[2]);

			// clamp cell coordinates to the extent of the grid
			const std::uint32_t cx = static_cast<std::uint32_t>((std::max)(0.0, (std::min)(double(last_cell_[0]), cell_x)));
			const std::uint32_t cy = static_cast<std::uint32_t>((std::max)(0.0, (std::min)(double(last_cell_[1]), cell_y)));
			const std::uint32_t cz = static_cast<std::uint32_t>((std::max)(0.0, (std::min)(double(last_cell_[2]), cell_z)));

			// combo of Morton key + outside-grid bit
			return (morton_key(cx, cy, cz) << 1) | std::uint64_t(outside_grid);
		}

		// spreads 21 coordinate bits into bit positions 0, 3, 6, ..., 60 of a 63-bit key
		static std::uint64_t spread_bits(std::uint32_t value)
		{
			std::uint64_t bits = value;

			bits = (bits | (bits << 32)) & 0x1f00000000ffffULL;
			bits = (bits | (bits << 16)) & 0x1f0000ff0000ffULL;
			bits = (bits | (bits << 8)) & 0x100f00f00f00f00fULL;
			bits = (bits | (bits << 4)) & 0x10c30c30c30c30c3ULL;
			bits = (bits | (bits << 2)) & 0x1249249249249249ULL;

			return bits;
		}

		// interleaves three 21-bit coordinates into a single 63-bit Morton key
		// 'x' occupies bit positions: 0, 3, 6, ..., 60
		// 'y' occupies bit positions: 1, 4, 7, ..., 61
		// 'z' occupies bit positions: 2, 5, 8, ..., 62
		static std::uint64_t morton_key(std::uint32_t x, std::uint32_t y, std::uint32_t z)
		{
#if (defined(_M_X64) || defined(__x86_64__)) && !defined(_M_ARM64EC)
			// spread x and y together in the two 64-bit SSE2 lanes, z with scalar shifts
			__m128i xy = _mm_set_epi64x(y, x);
			xy = _mm_and_si128(_mm_or_si128(xy, _mm_slli_epi64(xy, 32)), _mm_set1_epi64x(0x1f00000000ffffULL));
			xy = _mm_and_si128(_mm_or_si128(xy, _mm_slli_epi64(xy, 16)), _mm_set1_epi64x(0x1f0000ff0000ffULL));
			xy = _mm_and_si128(_mm_or_si128(xy, _mm_slli_epi64(xy, 8)), _mm_set1_epi64x(0x100f00f00f00f00fULL));
			xy = _mm_and_si128(_mm_or_si128(xy, _mm_slli_epi64(xy, 4)), _mm_set1_epi64x(0x10c30c30c30c30c3ULL));
			xy = _mm_and_si128(_mm_or_si128(xy, _mm_slli_epi64(xy, 2)), _mm_set1_epi64x(0x1249249249249249ULL));
			const std::uint64_t spread_x = static_cast<std::uint64_t>(_mm_cvtsi128_si64(xy));
			const std::uint64_t spread_y = static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(xy, 8)));
			return spread_x | (spread_y << 1) | (spread_bits(z) << 2);
#else
			// fallback, spreading each coordinate separately and combining them
			return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2);
#endif
		}
	};
}

CELLMATE_DETAIL_FP_END
