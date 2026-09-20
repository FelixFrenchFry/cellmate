#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace cellmate_test
{
	using std::uint32_t;

	struct PointCloud
	{
		std::vector<float> x, y, z;
	};

	// generate points in [min, max) along each axis
	inline PointCloud make_uniform_cloud(std::size_t N_points, uint32_t seed, float min, float max)
	{
		if (!std::isfinite(min) || !std::isfinite(max) || min >= max)
		{
			throw std::invalid_argument("make_uniform_cloud() bounds must be finite and min < max");
		}

		PointCloud points;
		points.x.resize(N_points);
		points.y.resize(N_points);
		points.z.resize(N_points);

		std::mt19937 random(seed);
		const double extent = double(max) - min;
		const float last = std::nextafter(max, min);
		
		for (std::size_t i = 0; i < N_points; i++)
		{
			// using mt19937, upper 24 bits divided by 2^24, x/y/z order
			const float ux = float(random() >> 8) / 16777216.0f;
			const float uy = float(random() >> 8) / 16777216.0f;
			const float uz = float(random() >> 8) / 16777216.0f;

			// keep the upper bound exclusive after float rounding
			points.x[i] = (std::min)(last, float(double(min) + extent * ux));
			points.y[i] = (std::min)(last, float(double(min) + extent * uy));
			points.z[i] = (std::min)(last, float(double(min) + extent * uz));
		}

		return points;
	}
}
