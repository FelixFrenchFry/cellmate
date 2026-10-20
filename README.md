[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![C++14](https://img.shields.io/badge/C%2B%2B-14-blue.svg)](https://en.cppreference.com/w/cpp/14) [![Tests](https://github.com/FelixFrenchFry/cellmate/actions/workflows/tests.yml/badge.svg)](https://github.com/FelixFrenchFry/cellmate/actions/workflows/tests.yml)

# 🔎 cellmate::NeighborSearch

Efficient fixed-radius neighbor search in 3D point clouds, dense or sparse, header-only for C++14 and later.

The class `cellmate::NeighborSearch` finds every point within a radius of a query position and returns their indices. Under the hood, cellmate is a compact cell-linked list (CLL) in Morton order [Mor66], related to the CLL variant by Band et al. [BGT20] and the compact hashing it builds on [IABT11]. While the CLL has to be rebuilt whenever points move, rebuilds are extremely cheap, and queries read contiguous memory. Together, this optimizes for high query speed across a wide variety of point counts, densities and geometries.

* **Time** - rebuilds take O(N) time for N points, using a radix sort on the Morton keys and a single pass to build the cells. A query costs O(log C + M) for C occupied cells and M candidate points in its surrounding cells, which is O(M) in practice thanks to a cell prefix directory and lookup cache. In comparison, tree-based data structures like k-d trees and octrees typically take O(N log N) to build and O(log N + M) per query.
* **Memory** - the compact data structure uses O(N) memory for N points, since only occupied cells are stored, independent of the size of the domain. Sparse or clustered point clouds therefore cost no more memory than a dense distribution of points.
* **Parallelism** - the rebuild and batch query pipelines are heavily parallelized using OpenMP directives. Additionally, each function has a sequential fallback version for builds without OpenMP or when this execution mode is specified.
* **Vectorization** - distance checks, cell keys, candidate gathering, ... and other hot loops are vectorized using SIMD with AVX2 or AVX-512 intrinsics, selected at compile time. Additionally, each operation has a scalar fallback version with bit-identical results for CPUs or builds without these instruction sets.

## References

* [BGT20] S. Band, C. Gissler and M. Teschner, 2020. "Compressed Neighbour Lists for SPH", Computer Graphics Forum 39, 1, 531-542.
* [IABT11] M. Ihmsen, N. Akinci, M. Becker and M. Teschner, 2011. "A Parallel SPH Implementation on Multi-Core CPUs", Computer Graphics Forum 30, 1, 99-112.
* [Mor66] G. M. Morton, 1966. "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing", Technical report, IBM Ltd., Ottawa.
