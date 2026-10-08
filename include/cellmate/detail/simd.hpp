#pragma once

// choose one backend for every translation unit that includes cellmate
#if (defined(CELLMATE_SIMD_AUTO) + defined(CELLMATE_SIMD_SCALAR) + defined(CELLMATE_SIMD_AVX256) + defined(CELLMATE_SIMD_AVX512)) > 1
#error "cellmate::simd backend selections must not conflict"
#endif

#if defined(CELLMATE_SIMD_AVX256) && !defined(__AVX__)
#error "cellmate::simd AVX256 requires an AVX compiler target"
#endif

#if defined(CELLMATE_SIMD_AVX512) && !defined(__AVX512F__)
#error "cellmate::simd AVX512 requires an AVX512F compiler target"
#endif

// AUTO selects AVX256 if the compiler target enables AVX, otherwise scalar
// (AVX512 is only selected when CELLMATE_SIMD_AVX512 is explicitly defined)
#if defined(CELLMATE_SIMD_AVX512)
#include "simd_avx512.hpp"

namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryAVX512;
	}
}
#elif !defined(CELLMATE_SIMD_SCALAR) && defined(__AVX__)
#include "simd_avx256.hpp"

namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryAVX256;
	}
}
#else
#include "simd_scalar.hpp"

namespace cellmate
{
	namespace detail
	{
		using DistQuery = DistQueryScalar;
	}
}
#endif
