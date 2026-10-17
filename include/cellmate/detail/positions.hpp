#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "fp_control.hpp"

// match backend settings so position reads can inline into the search loops
CELLMATE_DETAIL_FP_BEGIN

namespace cellmate
{
	namespace detail
	{
		// SoA input positions: (x, y, z) as 3 separate float arrays
		// (e.g. 3 std::vector<float>, 3 std::array<float, N>, 3 float[N] arrays, ...)
		struct Positions_SoA
		{
			const float* x_;
			const float* y_;
			const float* z_;

			bool valid(std::size_t) const noexcept
			{
				return x_ && y_ && z_;
			}

			std::array<float, 3> pos(std::size_t idx) const noexcept
			{
				return { x_[idx], y_[idx], z_[idx] };
			}

			float x(std::size_t idx) const noexcept { return x_[idx]; }
			float y(std::size_t idx) const noexcept { return y_[idx]; }
			float z(std::size_t idx) const noexcept { return z_[idx]; }
		};

		// AoS input positions: (x, y, z) as 3 consecutive floats, with the points 'stride_bytes_' apart
		// (e.g. std::array<float, 3>, glm::vec3, Eigen::Vector3f, cv::Point3f, pcl::PointXYZ, ...)
		struct Positions_AoS
		{
			const float* xyz_ = nullptr;
			std::size_t stride_bytes_ = 3 * sizeof(float);

			const float* point(std::size_t idx) const noexcept
			{
				return reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(xyz_) + idx * stride_bytes_);
			}

			bool valid(std::size_t N_points) const noexcept
			{
				return
					xyz_ && // not null
					stride_bytes_ >= 3 * sizeof(float) && // points do not overlap
					stride_bytes_ % alignof(float) == 0 && // every point's x is float-aligned
					(N_points == 0 || N_points - 1 <= (std::numeric_limits<std::size_t>::max)() / stride_bytes_); // last point's offset fits in 'size_t'
			}

			std::array<float, 3> pos(std::size_t idx) const noexcept
			{
				const float* p = point(idx);
				return { p[0], p[1], p[2] };
			}

			float x(std::size_t idx) const noexcept { return point(idx)[0]; }
			float y(std::size_t idx) const noexcept { return point(idx)[1]; }
			float z(std::size_t idx) const noexcept { return point(idx)[2]; }
		};
	}
}

CELLMATE_DETAIL_FP_END
