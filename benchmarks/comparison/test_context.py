"""Check comparison metadata with CMake, Ninja, Git and a compiler on PATH."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
MSVC = os.name == "nt"


class ContextTests(unittest.TestCase):
    def setUp(self):
        temporary_root = (REPOSITORY / "build").resolve()
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="context-", dir=temporary_root)
        self.root = Path(self.temp.name).resolve()
        self.root.relative_to(temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.source = self.root / "source"
        self.source.mkdir()
        self.build = self.root / "build"
        self.environment = os.environ.copy()
        self.environment["GIT_CEILING_DIRECTORIES"] = str(self.root)
        self.environment["GIT_CONFIG_NOSYSTEM"] = "1"
        self.environment["GIT_CONFIG_GLOBAL"] = os.devnull
        self.environment["GIT_OPTIONAL_LOCKS"] = "0"
        cmake = (HERE / "CMakeLists.txt").read_text(encoding="utf-8")
        marker = "# record the measured commit, compiler and flags in the benchmark context"
        context = cmake.split(marker, 1)[1].split("\nif(CELLMATE_COMPARISON_TREENSEARCH)", 1)[0]
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\nproject(cellmate LANGUAGES CXX)\n"
            "add_executable(cellmate_comparison main.cpp)\n" + context,
            encoding="utf-8",
        )
        shutil.copyfile(HERE / "ComparisonContext.cmake", self.source / "ComparisonContext.cmake")
        (self.source / "main.cpp").write_text(
            '#include <cstdio>\n#include "comparison_revision.hpp"\n'
            'int main() { std::puts(CELLMATE_COMPARISON_COMMIT); '
            'std::puts(CELLMATE_COMPARISON_FLAGS); std::puts(CELLMATE_COMPARISON_CONFIG); }\n',
            encoding="utf-8",
        )
        self.command("git", "init", "-q")
        self.commit()

    def command(self, *arguments):
        result = subprocess.run(arguments, cwd=self.source, env=self.environment,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def commit(self):
        self.command("git", "add", ".")
        self.command("git", "-c", "user.name=Context Test", "-c",
                     "user.email=test@users.noreply.github.com", "-c", "commit.gpgsign=false",
                     "commit", "--no-verify", "-qm", "fixture")

    def configure(self, flags=""):
        arguments = ["cmake", "-S", str(self.source), "-B", str(self.build),
                     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=" + flags]
        if MSVC:
            arguments.append("-DCMAKE_CXX_FLAGS_DEBUG=/Od /Z7")
        self.command(*arguments)

    def run_build(self):
        self.command("cmake", "--build", str(self.build))
        executable = self.build / ("cellmate_comparison.exe" if MSVC else "cellmate_comparison")
        output = self.command(str(executable)).splitlines()
        return {"commit": output[0], "flags": output[1], "config": output[2]}, executable

    def test_flags_round_trip_quotes_backslashes_and_semicolons(self):
        include = self.source / "include path; #literal"
        include.mkdir()
        flags = ('/I"' if MSVC else '-I"') + str(include) + '"'
        self.configure(flags)
        metadata, _ = self.run_build()
        self.assertTrue(metadata["flags"].startswith(flags + " "), metadata)
        self.assertEqual(metadata["config"], "Release")

    def test_incremental_build_refreshes_commit_without_noop_recompile(self):
        self.configure()
        first, _ = self.run_build()
        self.assertEqual(first["commit"], self.command("git", "rev-parse", "--short=12", "HEAD"))
        with (self.source / "main.cpp").open("a", encoding="utf-8") as stream:
            stream.write("// tracked edit\n")
        edited, _ = self.run_build()
        self.assertEqual(edited["commit"], first["commit"])
        self.commit()
        committed, executable = self.run_build()
        self.assertEqual(committed["commit"], self.command("git", "rev-parse", "--short=12", "HEAD"))
        self.assertNotEqual(committed["commit"], first["commit"])
        header = self.build / "comparison_revision.hpp"
        mtimes = (header.stat().st_mtime_ns, executable.stat().st_mtime_ns)
        unchanged, _ = self.run_build()
        self.assertEqual(unchanged, committed)
        self.assertEqual(mtimes, (header.stat().st_mtime_ns, executable.stat().st_mtime_ns))

if __name__ == "__main__":
    unittest.main(verbosity=2)
