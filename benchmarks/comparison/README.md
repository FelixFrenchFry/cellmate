# Library comparison benchmarks

The comparison measures fixed-radius neighbor search in `cellmate::NeighborSearch` and eight other libraries. The benchmarks use the same points, queries, and search radius for each library, and check sampled neighbor lists against brute force. The [main README](../../README.md#benchmarks) shows the timing and peak memory results.

## Build and run

Build the opt-in comparison executable from the repository root. CMake fetches the pinned libraries and requires OpenMP. On a machine without AVX2, add `-DCELLMATE_COMPARISON_TREENSEARCH=OFF` to the configure command.

```sh
cmake -S . -B build/comparison -DCMAKE_BUILD_TYPE=Release -DCELLMATE_BUILD_COMPARISON=ON -DCELLMATE_ENABLE_OPENMP=ON
cmake --build build/comparison --config Release --target cellmate_comparison
```

In the examples below, replace `<executable>` with the path to the built `cellmate_comparison` (`cellmate_comparison.exe` on Windows).

```text
<executable> --benchmark_list_tests
<executable> comparison --benchmark_filter=/reference/
```

The executable defaults to `comparison`, which combines reference cases and scaling sweeps. The focused profiles vary the distribution (`comp_scenarios`), indexed point count (`comp_points`), query count (`comp_queries`), target average neighbors (`comp_neighbors`), or requested threads (`comp_threads`).

### Run profiles

Use Python 3.10 or later to run each case in its own process. The script runs the five focused profiles by default, with three repetitions and a 0.5-second minimum time. Repeat `--profile` to choose profiles or use `--filter` to select cases by name. Use `--repetitions` and `--minimum` to change the timing settings, `--timeout` to set a per-process limit in seconds, and `--hardware` to label the test system.

```text
python benchmarks/comparison/run_profiles.py <executable> build/benchmark-results/comparison --timeout 900
```

The runner saves benchmark JSON, logs, `run-status.json`, and `memory.csv`. It continues after failed cases and exits with status 1 if any fail or no cases run. On Windows, `memory.csv` records peak working set and peak private commit in bytes when available.

Run `python -m unittest discover -s benchmarks/comparison` to test the tools. The context tests require CMake, Ninja, Git, and a C++ compiler.

## Libraries

| Library | Revision | License | Index | Input positions |
|---|---|---|---|---|
| cellmate | this repository | MIT | compact cell-linked list | copied (`cpy`) or referenced (`ref`) |
| [nanoflann](https://github.com/jlblancoc/nanoflann) | `v1.10.0` | BSD-2 | k-d tree | referenced |
| [PicoTree](https://github.com/Jaybro/pico_tree) | `v1.0.0` | MIT | k-d tree | referenced |
| [unibn Octree](https://github.com/jbehley/octree) | `8e3927d48d5ce61f94aab6090d77b4434f87dc89` | MIT | octree | referenced |
| [Ponca](https://github.com/poncateam/ponca) | `v1.4` | MPL-2.0 | k-d tree | copied |
| [CompactNSearch](https://github.com/InteractiveComputerGraphics/CompactNSearch) | `b40afcf47fe1963b363eba2371f04b42720fcb1d` | MIT | compact hash grid | referenced |
| [CGAL](https://www.cgal.org) | `5.6.2` | GPL-3.0+ (Spatial Searching) | k-d tree | copied |
| [Boost.Geometry R-tree](https://www.boost.org/libs/geometry) | `1.86.0` | BSL-1.0 | R*-tree | copied |
| [TreeNSearch](https://github.com/InteractiveComputerGraphics/TreeNSearch) | `61cf111f65c01f1d71fb5023ac72ce6bb0e8d6f9` | MIT | octree | referenced |

[CMakeLists.txt](CMakeLists.txt) pins the download URLs and SHA-256 hashes. Input ownership in the table describes what the index stores. Adapters that need a different input layout convert the shared x, y, and z arrays outside timing.

## What gets measured

| Operation | Timed work |
|---|---|
| `rebuild` | Rebuild the index, reusing the library object where possible. |
| `query` | Run one query per position on the calling thread and reuse the output buffer. |
| `query_batch` | Query all positions on an existing index and retain the results for the whole batch. |
| `rebuild_query_batch` | Rebuild the index and run the batch, as in a simulation step with moving points. |

The benchmark warms up reusable storage and checks sampled results before and after timing. Radius calibration places the cutoff between distinct neighbor distances. A mismatch still fails the case. Timing plots show the mean across repetitions, with bar whiskers extending one sample standard deviation on either side. Memory plots show peak private commit for the whole process, including setup, warmup, the index, and retained results.

cellmate, CompactNSearch, and TreeNSearch use their own batch APIs. The other adapters run single queries in a shared OpenMP loop and merge the per-thread buffers into a flat neighbor array with offsets. CompactNSearch and TreeNSearch keep their own neighbor lists and use separate query and data point sets to include self matches.

## Adapter settings

* **cellmate** - runs sequentially with one requested thread and uses the parallel path otherwise. Batch queries use `QueryOrder::morton`.
* **nanoflann, PicoTree, unibn Octree, and Ponca** - use strict radius cutoffs. Their leaf or bucket sizes are 10, 10, 32, and 64, respectively. nanoflann disables sorting of query results and uses `n_thread_build` for parallel rebuilds.
* **CompactNSearch** - forces a full hash rebuild, with part of the rebuild parallelized. It z-sorts its input copy once outside timing and maps the resulting indices back for validation.
* **CGAL and Boost.Geometry** - CGAL explicitly builds a `Kd_tree` and searches it with a zero-fuzziness sphere. Boost bulk-loads an `rstar<16>` tree and filters the points in the sphere's bounding box by distance.
* **TreeNSearch** - measures `rebuild_query_batch` only because `run()` builds and queries the index together. The adapter uses the requested thread count and leaves optional z-sorting disabled.

The [adapters](adapters/) contain the exact library calls and parameters. The [profiles](profiles/) define the workloads, seeds, and requested thread counts.
