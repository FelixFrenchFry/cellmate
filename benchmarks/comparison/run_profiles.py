"""Run each comparison case in its own process. Log failures and record peak memory on Windows."""

import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


PROFILES = ("comp_scenarios", "comp_threads", "comp_neighbors", "comp_queries", "comp_points")
OOM = re.compile(
    r"\b(?:bad_alloc|out of memory|cannot allocate memory|not enough memory|paging file is too small)\b",
    re.IGNORECASE,
)
DIAGNOSTIC = re.compile(r"\b(?:error|fatal|exception|too many|assertion|abort(?:ed)?)\b", re.IGNORECASE)


if os.name == "nt":
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    get_process_memory = kernel32.K32GetProcessMemoryInfo
    get_process_memory.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD
    )
    get_process_memory.restype = wintypes.BOOL
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def read_peak_memory(pid):
    if os.name != "nt":
        return None, None

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None, None
    try:
        counters = ProcessMemoryCounters(cb=ctypes.sizeof(ProcessMemoryCounters))
        if get_process_memory(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize, counters.PeakPagefileUsage
        return None, None
    finally:
        kernel32.CloseHandle(handle)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_command(command, timeout, log, measure_memory=False):
    error = None
    peaks = (None, None)
    try:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors="replace"
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                error = {"reason": "timeout", "message": f"Exceeded the {timeout:g}-second timeout",
                         "exit_code": None}
            if measure_memory:
                # Popen still holds the original process handle after the child exits.
                peaks = read_peak_memory(process.pid)
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except OSError as exception:
        reason = "out_of_memory" if exception.errno == errno.ENOMEM else "launch_error"
        print(str(exception), file=log)
        return None, {"reason": reason, "message": str(exception), "exit_code": None}, peaks
    log.write(stdout + stderr)
    if error:
        print(error["message"], file=log)
    return result, error, peaks


def process_failure(result, benchmark_message=None):
    lines = (result.stderr + "\n" + result.stdout).splitlines()
    oom_message = next((line.strip() for line in lines if OOM.search(line)), None)
    if benchmark_message and OOM.search(benchmark_message):
        oom_message = benchmark_message
    # Windows STATUS_NO_MEMORY and STATUS_COMMITMENT_LIMIT.
    oom_code = os.name == "nt" and (result.returncode & 0xFFFFFFFF) in (0xC0000017, 0xC000012D)
    exit_message = f"Process exited with code {result.returncode}"
    if oom_message or oom_code:
        return {"reason": "out_of_memory", "message": oom_message or exit_message, "exit_code": result.returncode}
    if benchmark_message:
        return {"reason": "benchmark_error", "message": benchmark_message, "exit_code": result.returncode}
    if result.returncode:
        message = next((line.strip() for line in lines if DIAGNOSTIC.search(line)), exit_message)
        return {"reason": "process_exit", "message": message, "exit_code": result.returncode}
    return None


def read_case(path, case, repetitions, result):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["benchmarks"]
        if not isinstance(data["context"], dict) or not isinstance(rows, list):
            raise ValueError("Expected benchmark context and rows")
        if not all(isinstance(row, dict) and row.get("run_name") == case for row in rows):
            raise ValueError("Output contains a different case")
        benchmark_message = next(
            (row.get("error_message") or "Benchmark reported an error"
             for row in rows if row.get("error_occurred")), None
        )
        error = process_failure(result, benchmark_message)
        if error:
            return None, error
        iterations = [row for row in rows if row.get("run_type") == "iteration"]
        indices = sorted(row["repetition_index"] for row in iterations)
        if indices != list(range(repetitions)):
            raise ValueError(f"Expected {repetitions} repetitions, got indices {indices}")
        times = [row["real_time"] for row in iterations]
        if any(row["time_unit"] != "ms" for row in iterations) or any(
            not math.isfinite(value) or value <= 0 for value in times
        ):
            raise ValueError("Expected positive, finite repetition times in milliseconds")
        if repetitions > 1:
            means = [row["real_time"] for row in rows if row.get("aggregate_name") == "mean"]
            if len(means) != 1 or not math.isclose(means[0], math.fsum(times) / repetitions, rel_tol=1e-10):
                raise ValueError("Expected one mean matching the repetition times")
        return data, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, process_failure(result) or {
            "reason": "invalid_result", "message": str(error), "exit_code": result.returncode
        }


def run_profile(command, output, filter_pattern, repetitions, minimum, timeout, memory):
    profile = output.stem
    cases, errors = [], []
    completed = 0
    combined = {"context": {}, "benchmarks": []}
    filters = [f"--benchmark_filter={filter_pattern}"] if filter_pattern else []
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        result, error, _ = run_command([*command, "--benchmark_list_tests", *filters], timeout, log)
        if not error:
            error = process_failure(result)
        if error:
            errors.append({"stage": "discovery", **error})
        else:
            cases = result.stdout.splitlines()
            if len(cases) != len(set(cases)):
                errors.append({"stage": "discovery", "reason": "invalid_result",
                               "message": "The executable listed duplicate case names"})
                cases = []

        print(f"{profile}: running {len(cases)} cases", flush=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
            raw = Path(scratch) / "case.json"
            for index, case in enumerate(cases, start=1):
                raw.unlink(missing_ok=True)
                print(f"\n[{index}/{len(cases)}] {case}", file=log, flush=True)
                result, error, peaks = run_command([
                    *command, f"--benchmark_filter=^{re.escape(case)}$",
                    f"--benchmark_out={raw}", "--benchmark_out_format=json",
                    f"--benchmark_repetitions={repetitions}", f"--benchmark_min_time={minimum}",
                    "--benchmark_report_aggregates_only=false",
                ], timeout, log, measure_memory=True)
                if not error:
                    data, error = read_case(raw, case, repetitions, result)
                memory.writerow((profile, case, "failed" if error else "complete", *peaks))
                if error:
                    errors.append({"case": case, **error})
                    print(f"{profile}: {case} failed ({error['reason']})", flush=True)
                    continue
                if not completed:
                    combined["context"] = data["context"]
                combined["benchmarks"].extend(data["benchmarks"])
                completed += 1

    write_json(output, combined)
    print(f"{profile}: {completed}/{len(cases)} complete", flush=True)
    return {
        "status": "failed" if errors else "complete",
        "expected_cases": len(cases),
        "recorded_cases": completed,
        "incomplete_cases": [error["case"] for error in errors if "case" in error],
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="Built cellmate_comparison executable")
    parser.add_argument("output", type=Path, help="New directory for profile JSON, logs, run-status.json, and memory.csv")
    parser.add_argument("--profile", action="append", choices=(*PROFILES, "comparison"),
                        help="Profile to run. Repeat to select several. Defaults to the five comparison profiles.")
    parser.add_argument("--filter", default="", help="Google Benchmark regex to select cases")
    parser.add_argument("--repetitions", type=int, default=3, choices=range(1, 11), metavar="1-10")
    parser.add_argument("--minimum", default="0.5s", help="Google Benchmark minimum time or iteration count")
    parser.add_argument("--timeout", type=float, help="Optional process timeout in seconds")
    parser.add_argument("--hardware", help="Test-system description to include in the report")
    args = parser.parse_args()
    if args.timeout is not None and not (math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("--timeout must be a positive finite number")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Executable not found: {executable}")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(str(error))

    status_path = output / "run-status.json"
    status = {
        "executable": str(executable),
        "exe_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "started": datetime.now().astimezone().isoformat(),
        "repetitions": args.repetitions,
        "minimum": args.minimum,
        "filter": args.filter,
        "timeout_seconds": args.timeout,
        "profiles": {},
    }
    if args.hardware:
        status["hardware"] = args.hardware
    profiles = status["profiles"]
    with (output / "memory.csv").open("w", encoding="utf-8", newline="", buffering=1) as memory_file:
        memory = csv.writer(memory_file)
        memory.writerow(("profile", "case", "status", "peak_working_set_bytes", "peak_private_commit_bytes"))
        for profile in dict.fromkeys(args.profile or PROFILES):
            profiles[profile] = {"status": "running"}
            write_json(status_path, status)
            profiles[profile] = run_profile(
                [str(executable), profile], output / f"{profile}.json", args.filter,
                args.repetitions, args.minimum, args.timeout, memory,
            )
            write_json(status_path, status)
    status["finished"] = datetime.now().astimezone().isoformat()
    write_json(status_path, status)
    if not any(profile["expected_cases"] for profile in profiles.values()):
        print("No benchmark cases ran")
        return 1
    return int(any(profile["errors"] for profile in profiles.values()))


if __name__ == "__main__":
    raise SystemExit(main())
