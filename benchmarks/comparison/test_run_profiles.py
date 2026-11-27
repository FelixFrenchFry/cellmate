"""Test how the comparison runner handles failed and incomplete cases."""

import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from run_profiles import run_profile


class RunProfileTests(unittest.TestCase):
    def test_failures_leave_later_cases_and_diagnostics_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "benchmark.py"
            fixture.write_text(
                """
import json
from pathlib import Path
import re
import sys
import time

cases = (
    "suite/first.1", "suite/firstX1", "suite/fatal", "suite/allocate",
    "suite/rejected", "suite/incomplete", "suite/bad_timing", "suite/hang", "suite/last",
)
args = sys.argv[1:]
filter_arg = next((arg for arg in args if arg.startswith("--benchmark_filter=")), None)
pattern = filter_arg.split("=", 1)[1] if filter_arg else "."
selected = [case for case in cases if re.search(pattern, case)]
if "--benchmark_list_tests" in args:
    print(*selected, sep="\\n")
    sys.exit(0)
assert len(selected) == 1, selected
case = selected[0]
output = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--benchmark_out=")))
if case == "suite/fatal":
    allocation = bytearray(b"x") * (64 * 1024 * 1024)
    del allocation
    output.write_text('{"benchmarks": [', encoding="utf-8")
    print("Fatal: candidate buffer limit exceeded")
    sys.exit(91)
if case == "suite/allocate":
    print("std::bad_alloc", file=sys.stderr)
    sys.exit(1)
if case == "suite/hang":
    time.sleep(10)
rows = [
    {"run_name": case, "run_type": "iteration", "repetition_index": index,
     "real_time": -1 if case == "suite/bad_timing" else 1, "time_unit": "ms"}
    for index in range(1 if case == "suite/incomplete" else 2)
]
if case == "suite/rejected":
    rows = [{"run_name": case, "error_occurred": True, "error_message": "wrong neighbor IDs"}]
else:
    rows.append({"run_name": case, "run_type": "aggregate", "aggregate_name": "mean",
                 "real_time": 1, "time_unit": "ms"})
output.write_text(json.dumps({"context": {"profile": "fixture"}, "benchmarks": rows}), encoding="utf-8")
sys.exit(1 if case == "suite/rejected" else 0)
""",
                encoding="utf-8",
            )
            output = root / "fixture.json"
            memory_log = io.StringIO(newline="")
            status = run_profile(
                [sys.executable, str(fixture)], output, "", repetitions=2, minimum="1x", timeout=2,
                memory=csv.writer(memory_log),
            )

            self.assertEqual(status["expected_cases"], 9)
            self.assertEqual(status["recorded_cases"], 3)
            self.assertEqual(status["status"], "failed")
            errors = {entry["case"]: entry for entry in status["errors"]}
            self.assertEqual(errors["suite/fatal"]["reason"], "process_exit")
            self.assertEqual(errors["suite/fatal"]["exit_code"], 91)
            self.assertIn("candidate buffer limit", errors["suite/fatal"]["message"])
            self.assertEqual(errors["suite/allocate"]["reason"], "out_of_memory")
            self.assertEqual(errors["suite/rejected"]["message"], "wrong neighbor IDs")
            self.assertEqual(errors["suite/incomplete"]["reason"], "invalid_result")
            self.assertEqual(errors["suite/bad_timing"]["reason"], "invalid_result")
            self.assertEqual(errors["suite/hang"]["reason"], "timeout")
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["context"], {"profile": "fixture"})
            self.assertEqual(
                [row["run_name"] for row in result["benchmarks"]],
                ["suite/first.1"] * 3 + ["suite/firstX1"] * 3 + ["suite/last"] * 3,
            )
            self.assertIn("std::bad_alloc", output.with_suffix(".log").read_text(encoding="utf-8"))

            memory_log.seek(0)
            rows = list(csv.reader(memory_log))
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row[0] == "fixture" for row in rows))
            memory = {case: (state, working_set, commit) for _, case, state, working_set, commit in rows}
            self.assertEqual(memory.keys(), set(errors) | {"suite/first.1", "suite/firstX1", "suite/last"})
            for case, (state, working_set, commit) in memory.items():
                self.assertEqual(state, "failed" if case in errors else "complete")
                if os.name == "nt":
                    self.assertGreater(int(working_set), 0)
                    self.assertGreater(int(commit), 0)
                else:
                    self.assertEqual((working_set, commit), ("", ""))
            if os.name == "nt":
                for metric in (1, 2):
                    self.assertGreater(
                        int(memory["suite/fatal"][metric]) - int(memory["suite/last"][metric]), 32 * 1024**2
                    )


if __name__ == "__main__":
    unittest.main()
