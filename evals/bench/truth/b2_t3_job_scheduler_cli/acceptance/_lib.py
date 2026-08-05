"""Shared helpers for b2_t3_job_scheduler_cli acceptance probes.

Not itself a probe: its filename does not match ``probe_<band>_<slug>.py``,
so it is never discovered or invoked directly by the grader. Every probe in
this directory imports it via ``sys.path.insert(0, os.path.dirname(__file__))``.

Stdlib-only, per the truth-directory contract.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile


def copy_tree(tree_path):
    """Copy the graded tree into a fresh temp dir; return the new path.

    Probes must never execute the CLI against the tree they were handed —
    only against this copy.
    """
    dest = tempfile.mkdtemp(prefix="b2-probe-")
    for name in os.listdir(tree_path):
        src = os.path.join(tree_path, name)
        dst = os.path.join(dest, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return dest


def run_cli(work_dir, args, timeout=60):
    """Run ``python3 jobsched.py <args>`` inside work_dir."""
    cmd = [sys.executable, "jobsched.py"] + list(args)
    return subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def write_file(work_dir, filename, content):
    path = os.path.join(work_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def read_state(work_dir, state_file=".jobsched/state.json"):
    path = os.path.join(work_dir, state_file)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_log(work_dir, log_file=".jobsched/logs/run.log"):
    path = os.path.join(work_dir, log_file)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def snapshot_paths(work_dir):
    """Sorted list of every relative path under work_dir (files and dirs)."""
    out = []
    for root, dirs, files in os.walk(work_dir):
        for name in dirs + files:
            full = os.path.join(root, name)
            out.append(os.path.relpath(full, work_dir))
    return sorted(out)


def snapshot_hashes(work_dir):
    """Map of every relative file path under work_dir to its sha256 hex
    digest. Stricter than snapshot_paths: catches content mutation of a
    file that was already present, not just files being added/removed.
    """
    out = {}
    for root, dirs, files in os.walk(work_dir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, work_dir)
            with open(full, "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg="ok"):
    print(f"PASS: {msg}")
    sys.exit(0)
