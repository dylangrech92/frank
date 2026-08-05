"""Edge: when two due jobs share the exact same priority, the tie is broken
by job_id ascending (lexicographic) — distinct from the core probe covering
unequal priorities.

Requirement 3, step 2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(
        work,
        "jobs.txt",
        "zeta  | 5 | 1 | - | true\n"
        "alpha | 5 | 1 | - | true\n"
        "mu    | 5 | 1 | - | true\n",
    )
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    lines = [l for l in log.splitlines() if l]
    job_order = [l.split("\t")[1] for l in lines]
    if job_order != ["alpha", "mu", "zeta"]:
        _lib.fail(f"expected job_id-ascending tie-break order, got {job_order}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
