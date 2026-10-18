"""Build scalar/AVX256/AVX512 tests, run first two by default, needs AVX2 x86/x64, CMake, and a C++ compiler"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-avx512", action="store_true",
                        help="Run AVX512 tests too (requires a compatible CPU and operating system)")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    windows = sys.platform == "win32"
    generator = "NMake Makefiles" if windows else "Unix Makefiles"
    executable_name = "cellmate_tests.exe" if windows else "cellmate_tests"
    backends = [
        ("SCALAR", ""),
        ("AVX256", "/arch:AVX2" if windows else "-mavx2"),
        ("AVX512", "/arch:AVX512" if windows else "-mavx512f"),
    ]

    def run(*command):
        subprocess.run(command, cwd=repository, check=True)

    try:
        for backend, flags in backends:
            build_directory = repository / "build" / "test-backends" / backend.lower()
            print(f"\nBuilding {backend} tests", flush=True)
            run(
                "cmake", "-S", str(repository), "-B", str(build_directory), "-G", generator,
                "-DCMAKE_BUILD_TYPE=Release", "-DCELLMATE_BUILD_TESTS=ON",
                "-DCELLMATE_BUILD_BENCHMARKS=OFF",
                "-DCELLMATE_ENABLE_OPENMP=OFF", f"-DCELLMATE_SIMD={backend}",
                f"-DCELLMATE_TEST_ARCH_FLAGS={flags}",
            )
            run(
                "cmake", "--build", str(build_directory), "--config", "Release",
                "--target", "cellmate_tests",
            )

            if backend == "AVX512" and not args.run_avx512:
                print("AVX512 tests built. Use '--run-avx512' to run them.", flush=True)
                continue

            print(f"Running {backend} tests", flush=True)
            run(str(build_directory / "tests" / executable_name))
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Backend tests failed: {error}\n")


if __name__ == "__main__":
    main()
