#pragma once

#include <cfloat>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// choose one backend for every translation unit that includes cellmate
#if (defined(CELLMATE_SIMD_AUTO) + defined(CELLMATE_SIMD_SCALAR) + defined(CELLMATE_SIMD_AVX256) + defined(CELLMATE_SIMD_AVX512)) > 1
#error "cellmate::simd backend selections must not conflict"
#endif

#if defined(CELLMATE_SIMD_AVX256) && !defined(__AVX2__)
#error "cellmate::simd AVX256 requires an AVX2 compiler target"
#endif

#if defined(CELLMATE_SIMD_AVX512) && !defined(__AVX512F__)
#error "cellmate::simd AVX512 requires an AVX512F compiler target"
#endif

#if (defined(CELLMATE_SIMD_AVX256) || defined(CELLMATE_SIMD_AVX512)) && (FLT_EVAL_METHOD != 0 && FLT_EVAL_METHOD != 1)
#error "cellmate::simd requires double evaluation precision, use SCALAR or -mfpmath=sse on GCC x86"
#endif

#if (defined(CELLMATE_SIMD_AVX256) || defined(CELLMATE_SIMD_AVX512)) && defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1930 && !defined(_M_FP_STRICT)
#error "cellmate::simd requires /fp:strict on MSVC before Visual Studio 2022, use SCALAR or enable /fp:strict"
#endif

// AUTO selects AVX256 only when AVX2 is available and scalar precision and contraction controls match
// (AVX512 is only selected when CELLMATE_SIMD_AVX512 is explicitly defined)
#if defined(CELLMATE_SIMD_AVX512)
#define CELLMATE_DETAIL_SIMD_WIDTH 512
#include "simd_avx512.hpp"

namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryAVX512;
	}
}
#elif !defined(CELLMATE_SIMD_SCALAR) && defined(__AVX2__) && (FLT_EVAL_METHOD == 0 || FLT_EVAL_METHOD == 1) && \
	(!defined(_MSC_VER) || defined(__clang__) || _MSC_VER >= 1930 || defined(_M_FP_STRICT))
#define CELLMATE_DETAIL_SIMD_WIDTH 256
#include "simd_avx256.hpp"

namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryAVX256;
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
	}
}
#endif

namespace cellmate
{
	namespace detail
	{
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
