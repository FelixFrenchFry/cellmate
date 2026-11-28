#pragma once

#include <cfloat>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// choose one backend for every translation unit that includes cellmate
#if (defined(CELLMATE_SIMD_AUTO) + defined(CELLMATE_SIMD_SCALAR) + defined(CELLMATE_SIMD_NEON64) + defined(CELLMATE_SIMD_AVX256) + defined(CELLMATE_SIMD_AVX512)) > 1
#error "cellmate::simd backend selections must not conflict"
#endif

#if defined(CELLMATE_SIMD_NEON64) && !((defined(__aarch64__) && defined(__ARM_NEON)) || (defined(_M_ARM64) && !defined(__clang__)))
#error "cellmate::simd 'NEON64' requires an ARM64 compiler target with NEON enabled"
#endif

#if defined(CELLMATE_SIMD_AVX256) && !defined(__AVX2__)
#error "cellmate::simd 'AVX256' requires an AVX2 compiler target"
#endif

#if defined(CELLMATE_SIMD_AVX512) && !defined(__AVX512F__)
#error "cellmate::simd 'AVX512' requires an AVX512F compiler target"
#endif

#if FLT_EVAL_METHOD != 0
#error "cellmate requires float evaluation precision ('FLT_EVAL_METHOD' == 0), use '-mfpmath=sse' on GCC x86"
#endif

// older MSVC may contract distance checks into FMA on AVX2 and NEON64 builds
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1930 && !defined(_M_FP_STRICT) && \
	(defined(__AVX2__) || defined(CELLMATE_SIMD_NEON64))
#error "cellmate requires '/fp:strict' on MSVC before Visual Studio 2022 when targeting AVX2 or selecting the NEON64 backend"
#endif

// 'AUTO' selects 'AVX256' whenever AVX2 is available
// ('NEON64' and 'AVX512' require explicit selection)
#if defined(CELLMATE_SIMD_AVX512)
#define CELLMATE_DETAIL_SIMD_WIDTH 512
#include "simd_avx512.hpp"
namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryAVX512;
		using DistQueryGroup = DistQueryGroupAVX512;
		using DistBox = DistBoxAVX512;
	}
}
#elif defined(CELLMATE_SIMD_NEON64)
#define CELLMATE_DETAIL_SIMD_WIDTH 128
#include "simd_neon64.hpp"
namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryNEON64;
		using DistQueryGroup = DistQueryGroupNEON64;
		using DistBox = DistBoxNEON64;
	}
}
#elif !defined(CELLMATE_SIMD_SCALAR) && defined(__AVX2__)
#define CELLMATE_DETAIL_SIMD_WIDTH 256
#include "simd_avx256.hpp"
namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryAVX256;
		using DistQueryGroup = DistQueryGroupAVX256;
		using DistBox = DistBoxAVX256;
	}
}
#else
#define CELLMATE_DETAIL_SIMD_WIDTH 0
#include "simd_scalar.hpp"
namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryScalar;
		using DistQueryGroup = DistQueryGroupScalar;
		using DistBox = DistBoxScalar;
	}
}
#endif

namespace cellmate
{
	namespace detail
	{
		constexpr bool is_scalar_backend = DistQuery::N_lanes == 1;

		// index of the lowest set bit in a nonzero mask
		inline unsigned first_set_bit(unsigned mask) noexcept
		{
#if defined(_MSC_VER)
			unsigned long lane;
			_BitScanForward(&lane, mask);
			return static_cast<unsigned>(lane);
#elif defined(__GNUC__) || defined(__clang__)
			return static_cast<unsigned>(__builtin_ctz(mask));
#else
			unsigned lane = 0;
			while ((mask & 1u) == 0)
			{
				mask >>= 1;
				lane++;
			}
			return lane;
#endif
		}
	}
}
