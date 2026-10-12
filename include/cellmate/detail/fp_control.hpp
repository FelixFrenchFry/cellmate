#pragma once

// detect x86 FMA in the compiler target, without runtime CPU detection
// (MSVC includes FMA in AVX2/AVX512 targets, while GCC and Clang expose __FMA__)
#if defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__) && (defined(__AVX2__) || defined(__AVX512F__)))
#define CELLMATE_DETAIL_HAS_FMA 1
#else
#define CELLMATE_DETAIL_HAS_FMA 0
#endif

// keep rounding controls local to cellmate implementation headers
// (leave consumer exception and floating-point environment settings unchanged)
#if defined(__clang__)
#define CELLMATE_DETAIL_FP_BEGIN \
	_Pragma("float_control(push)") \
	_Pragma("clang fp contract(off) exceptions(strict)")
#define CELLMATE_DETAIL_FP_END _Pragma("float_control(pop)")
#elif defined(_MSC_VER) && _MSC_VER >= 1930
#define CELLMATE_DETAIL_FP_BEGIN \
	__pragma(float_control(push)) \
	__pragma(float_control(precise, on)) \
	__pragma(fp_contract(off))
#define CELLMATE_DETAIL_FP_END __pragma(float_control(pop))
#elif defined(__GNUC__)
#define CELLMATE_DETAIL_FP_BEGIN \
	_Pragma("GCC push_options") \
	_Pragma("GCC optimize (\"fp-contract=off\")")
#define CELLMATE_DETAIL_FP_END _Pragma("GCC pop_options")
#else
// older MSVC requires caller-selected /fp:strict for SIMD
// (its float_control stack does not save and restore contraction settings)
#define CELLMATE_DETAIL_FP_BEGIN
#define CELLMATE_DETAIL_FP_END
#endif

// Clang restores floating-point settings at the end of each function body
// (strict exception semantics prevent contraction even with -ffp-contract=fast)
#if defined(__clang__)
#define CELLMATE_DETAIL_FP_NO_CONTRACT \
	_Pragma("clang fp contract(off) exceptions(strict)")
#else
#define CELLMATE_DETAIL_FP_NO_CONTRACT
#endif
