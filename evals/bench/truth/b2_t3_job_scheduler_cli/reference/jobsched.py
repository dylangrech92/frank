#!/usr/bin/env python3
"""jobsched — a dependency-aware, simulated-clock job scheduler CLI.

See README.md for the full command reference, schedule file format, and
exit-code contract. This module is stdlib-only by design.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

EXIT_OK = 0
EXIT_SCHEDULE_NOT_FOUND = 1
EXIT_PARSE_ERROR = 2
EXIT_DAG_ERROR = 3
EXIT_STATE_CORRUPT = 4
EXIT_STATE_MISMATCH = 5
EXIT_USAGE = 6
EXIT_INTERNAL = 70

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
STATE_VERSION = 1
STATE_REQUIRED_KEYS = (
    "version",
    "schedule_hash",
    "last_completed_tick",
    "in_progress_tick",
    "jobs",
)
LOG_ROTATE_THRESHOLD = 10000
LOG_MAX_ROTATIONS = 5


class UsageError(Exception):
    """A CLI usage problem: exit code 6."""


class ScheduleNotFound(Exception):
    """Schedule file missing/unreadable: exit code 1."""


class ParseError(Exception):
    def __init__(self, line_no, reason):
        super().__init__(f"line {line_no}: {reason}")
        self.line_no = line_no
        self.reason = reason


class DagError(Exception):
    """Unknown dependency or cycle: exit code 3."""


class StateCorrupt(Exception):
    def __init__(self, path):
        super().__init__(f"state file corrupt: {path} (use --force-recover to reset)")
        self.path = path


class StateMismatch(Exception):
    pass


# --------------------------------------------------------------------------
# Schedule parsing (Section 1)
# --------------------------------------------------------------------------

class Job:
    __slots__ = ("job_id", "priority", "interval", "depends_on", "command")

    def __init__(self, job_id, priority, interval, depends_on, command):
        self.job_id = job_id
        self.priority = priority
        self.interval = interval
        self.depends_on = depends_on
        self.command = command


def parse_schedule(path):
    """Return (jobs_by_id: dict[str, Job], order: list[str]).

    Raises ScheduleNotFound, ParseError, or DagError.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        raise ScheduleNotFound(path)

    jobs = {}
    order = []
    # Pass 1: validate rules 1-5 for every line, top to bottom, fail-fast on
    # the first violation. Duplicates (rule 6) are NOT checked here — per
    # spec, whole-file checks only begin once every line has individually
    # passed rules 1-5, so a later line's syntax error must be reported even
    # if an earlier line already duplicates another line's job_id.
    parsed_lines = []  # list of (line_no, job_id, priority, interval, depends_on, command)
    for line_no, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped[0] == "#":
            continue

        parts = raw.split("|", 4)
        if len(parts) != 5:
            raise ParseError(line_no, f"expected 5 fields separated by '|', got {len(parts)}")

        job_id = parts[0].strip()
        priority_raw = parts[1].strip()
        interval_raw = parts[2].strip()
        depends_raw = parts[3].strip()
        command = parts[4].strip()

        if not JOB_ID_RE.match(job_id):
            raise ParseError(line_no, f"invalid job id '{job_id}' (must match [A-Za-z0-9_-]+)")

        try:
            priority = int(priority_raw, 10)
        except ValueError:
            raise ParseError(line_no, "priority must be an integer")

        try:
            interval = int(interval_raw, 10)
        except ValueError:
            raise ParseError(line_no, "interval must be a positive integer")
        if interval < 1:
            raise ParseError(line_no, "interval must be a positive integer")

        if depends_raw == "-":
            depends_on = []
        else:
            depends_on = []
            for token in depends_raw.split(","):
                dep = token.strip()
                if not dep:
                    raise ParseError(line_no, "invalid depends_on list")
                depends_on.append(dep)

        parsed_lines.append((line_no, job_id, priority, interval, depends_on, command))

    # Pass 2 (whole-file, rule 6): duplicate job ids, first repeat wins,
    # reported at the later line's number.
    for line_no, job_id, priority, interval, depends_on, command in parsed_lines:
        if job_id in jobs:
            raise ParseError(line_no, f"duplicate job id '{job_id}'")
        jobs[job_id] = Job(job_id, priority, interval, depends_on, command)
        order.append(job_id)

    # Whole-file check: undefined dependency references (Section 1 rule 7 /
    # Section 2), scanning jobs top-down, each depends_on list left-to-right.
    for job_id in order:
        for dep in jobs[job_id].depends_on:
            if dep not in jobs:
                raise DagError(f"job '{job_id}' depends on unknown job '{dep}'")

    _check_acyclic(jobs, order)

    return jobs, order


def _check_acyclic(jobs, order):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {job_id: WHITE for job_id in order}

    def visit(job_id, stack):
        color[job_id] = GRAY
        stack.append(job_id)
        for dep in jobs[job_id].depends_on:
            if color[dep] == WHITE:
                cycle = visit(dep, stack)
                if cycle is not None:
                    return cycle
            elif color[dep] == GRAY:
                idx = stack.index(dep)
                return list(stack[idx:]) + [dep]
        stack.pop()
        color[job_id] = BLACK
        return None

    for job_id in order:
        if color[job_id] == WHITE:
            cycle = visit(job_id, [])
            if cycle is not None:
                _raise_cycle_error(cycle)


def _raise_cycle_error(cycle_ids):
    # cycle_ids is a closed walk [a, b, c, a] following depends_on edges.
    # Re-root it at the lexicographically smallest participating id.
    participants = cycle_ids[:-1]
    start = min(participants)
    start_idx = participants.index(start)
    rotated = participants[start_idx:] + participants[:start_idx] + [start]
    raise DagError("cycle detected: " + " -> ".join(rotated))


def count_edges(jobs):
    return sum(len(j.depends_on) for j in jobs.values())


# --------------------------------------------------------------------------
# State persistence (Section 5)
# --------------------------------------------------------------------------

def default_state_file():
    return os.path.join(".jobsched", "state.json")


def default_log_dir():
    return os.path.join(".jobsched", "logs")


def schedule_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fresh_state(sched_hash):
    return {
        "version": STATE_VERSION,
        "schedule_hash": sched_hash,
        "last_completed_tick": None,
        "in_progress_tick": None,
        "jobs": {},
    }


def load_state(state_file):
    """Return dict or None if no file exists. Raises StateCorrupt."""
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        raise StateCorrupt(state_file)
    if not isinstance(data, dict):
        raise StateCorrupt(state_file)
    for key in STATE_REQUIRED_KEYS:
        if key not in data:
            raise StateCorrupt(state_file)
    if not isinstance(data["jobs"], dict):
        raise StateCorrupt(state_file)
    return data


def write_state_atomic(state_file, state):
    directory = os.path.dirname(state_file) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".jobsched-state-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp_path, state_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# Run log (Section 6)
# --------------------------------------------------------------------------

def append_log_and_rotate(log_dir, lines):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
    if os.path.getsize(log_path) > LOG_ROTATE_THRESHOLD:
        _rotate_log(log_dir)


def _rotate_log(log_dir):
    oldest = os.path.join(log_dir, f"run.log.{LOG_MAX_ROTATIONS}")
    if os.path.exists(oldest):
        os.unlink(oldest)
    for n in range(LOG_MAX_ROTATIONS - 1, 0, -1):
        src = os.path.join(log_dir, f"run.log.{n}")
        dst = os.path.join(log_dir, f"run.log.{n + 1}")
        if os.path.exists(src):
            os.replace(src, dst)
    src = os.path.join(log_dir, "run.log")
    dst = os.path.join(log_dir, "run.log.1")
    os.replace(src, dst)


def log_line(tick, job_id, status, exit_code, reason):
    exit_field = "-" if exit_code is None else str(exit_code)
    reason_field = "-" if reason is None else reason
    return f"{tick}\t{job_id}\t{status}\t{exit_field}\t{reason_field}\n"


def existing_log_segments(log_dir):
    segments = []
    main = os.path.join(log_dir, "run.log")
    if os.path.exists(main):
        segments.append(main)
    for n in range(1, LOG_MAX_ROTATIONS + 1):
        seg = os.path.join(log_dir, f"run.log.{n}")
        if os.path.exists(seg):
            segments.append(seg)
    return segments


# --------------------------------------------------------------------------
# Scheduling core, shared by real run and dry-run (Section 3)
# --------------------------------------------------------------------------

def due_jobs_ordered(jobs, order, tick):
    due = [jobs[job_id] for job_id in order if tick % jobs[job_id].interval == 0]
    due.sort(key=lambda j: (-j.priority, j.job_id))
    return due


def is_eligible(job, completed_set):
    return all(dep in completed_set for dep in job.depends_on)


def completed_job_ids(state_jobs):
    """Ids whose most recent execution succeeded AND who have not since been
    cancelled (Section 10). Cancellation revokes a job's completed-ness for
    dependency-eligibility purposes from the moment it is recorded, even
    though `last_status` itself is left untouched as a historical record.
    """
    return {
        job_id
        for job_id, info in state_jobs.items()
        if info.get("last_status") == "completed" and not info.get("cancelled")
    }


def is_cancelled(state_jobs, job_id):
    info = state_jobs.get(job_id)
    return bool(info and info.get("cancelled"))


# --------------------------------------------------------------------------
# Command: validate
# --------------------------------------------------------------------------

def cmd_validate(args):
    jobs, order = parse_schedule(args.schedule_file)
    print(f"OK: {len(jobs)} jobs, {count_edges(jobs)} edges")
    return EXIT_OK


# --------------------------------------------------------------------------
# Command: run
# --------------------------------------------------------------------------

def cmd_run(args):
    jobs, order = parse_schedule(args.schedule_file)

    state_file = args.state_file
    log_dir = args.log_dir
    sched_hash = schedule_hash(args.schedule_file)

    try:
        state = load_state(state_file)
    except StateCorrupt:
        if not args.force_recover:
            raise
        state = None

    if state is not None and state["schedule_hash"] != sched_hash:
        if not args.force_recover:
            raise StateMismatch()
        state = None

    if state is None:
        state = fresh_state(sched_hash)

    if state["in_progress_tick"] is not None:
        start = state["in_progress_tick"]
    elif state["last_completed_tick"] is not None:
        start = state["last_completed_tick"] + 1
    else:
        start = 0

    n = args.until

    if start > n - 1:
        lct = state["last_completed_tick"]
        lct_text = "none" if lct is None else str(lct)
        print(f"already complete: last_completed_tick={lct_text}")
        return EXIT_OK

    dry_run = args.dry_run
    max_concurrent = args.max_concurrent

    if dry_run:
        # Entirely in-memory scratch copy; never touches disk. Cancellation
        # and the execution budget (Section 10 / Section 3 precedence) are
        # honored identically to a real run, so a multi-tick plan reflects
        # what a real run would actually do.
        completed = completed_job_ids(state["jobs"])
        for t in range(start, n):
            due = due_jobs_ordered(jobs, order, t)
            would_run = 0
            would_skip = 0
            budget_used = 0
            for job in due:
                if is_cancelled(state["jobs"], job.job_id):
                    would_skip += 1
                    continue
                if not is_eligible(job, completed):
                    would_skip += 1
                    continue
                if max_concurrent is not None and budget_used >= max_concurrent:
                    would_skip += 1
                    continue
                would_run += 1
                budget_used += 1
                completed.add(job.job_id)
            print(f"tick {t} (dry-run): {would_run} would-run, {would_skip} would-skip")
        print(f"done: {n} ticks planned (dry-run, no changes made)")
        return EXIT_OK

    completed = completed_job_ids(state["jobs"])

    for t in range(start, n):
        state["in_progress_tick"] = t
        write_state_atomic(state_file, state)

        due = due_jobs_ordered(jobs, order, t)
        lines = []
        c = f = s = 0
        budget_used = 0
        for job in due:
            # Precedence (Section 3 / Section 10): cancelled, THEN dependency
            # eligibility, THEN the execution budget. A job can match more
            # than one reason (e.g. cancelled AND dependency-unmet); only the
            # first matching reason is ever recorded.
            if is_cancelled(state["jobs"], job.job_id):
                s += 1
                lines.append(log_line(t, job.job_id, "skipped", None, "cancelled"))
                continue
            if not is_eligible(job, completed):
                s += 1
                lines.append(log_line(t, job.job_id, "skipped", None, "waiting-on-dependency"))
                continue
            if max_concurrent is not None and budget_used >= max_concurrent:
                s += 1
                lines.append(log_line(t, job.job_id, "skipped", None, "budget-exceeded"))
                continue

            proc = subprocess.run(
                job.command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            exit_code = proc.returncode
            status = "completed" if exit_code == 0 else "failed"
            info = state["jobs"].setdefault(
                job.job_id,
                {"run_count": 0, "last_run_tick": None, "last_status": None, "last_exit_code": None},
            )
            info["run_count"] += 1
            info["last_run_tick"] = t
            info["last_status"] = status
            info["last_exit_code"] = exit_code
            budget_used += 1
            if status == "completed":
                completed.add(job.job_id)
                c += 1
            else:
                completed.discard(job.job_id)
                f += 1
            lines.append(log_line(t, job.job_id, status, exit_code, None))

        if lines:
            append_log_and_rotate(log_dir, lines)

        state["last_completed_tick"] = t
        state["in_progress_tick"] = None
        write_state_atomic(state_file, state)

        print(f"tick {t}: {c} completed, {f} failed, {s} skipped")

    print(f"done: {n} ticks simulated")
    return EXIT_OK


# --------------------------------------------------------------------------
# Command: status
# --------------------------------------------------------------------------

def cmd_status(args):
    state_file = args.state_file
    if not os.path.exists(state_file):
        print("no state")
        return EXIT_OK

    try:
        state = load_state(state_file)
    except StateCorrupt:
        raise

    lct = state["last_completed_tick"]
    print(f"last_completed_tick: {'none' if lct is None else lct}")
    for job_id in sorted(state["jobs"].keys()):
        info = state["jobs"][job_id]
        # A cancelled job (Section 10) displays "cancelled" regardless of
        # whatever last_status it carries from before cancellation (or None,
        # if it was cancelled before ever executing). last_exit_code follows
        # the same "none" convention already used for last_completed_tick
        # when a job has never actually executed.
        display_status = "cancelled" if info.get("cancelled") else info["last_status"]
        exit_code = info["last_exit_code"]
        exit_display = "none" if exit_code is None else exit_code
        print(
            f"{job_id}: status={display_status} "
            f"run_count={info['run_count']} "
            f"last_exit_code={exit_display}"
        )
    return EXIT_OK


# --------------------------------------------------------------------------
# Command: cancel (Section 10)
# --------------------------------------------------------------------------

def cmd_cancel(args):
    jobs, order = parse_schedule(args.schedule_file)
    if args.job_id not in jobs:
        raise UsageError(f"cancel: unknown job id '{args.job_id}'")

    state_file = args.state_file
    sched_hash = schedule_hash(args.schedule_file)

    try:
        state = load_state(state_file)
    except StateCorrupt:
        if not args.force_recover:
            raise
        state = None

    if state is not None and state["schedule_hash"] != sched_hash:
        if not args.force_recover:
            raise StateMismatch()
        state = None

    if state is None:
        state = fresh_state(sched_hash)

    # cancel is the one operation besides Section 4's execution that may
    # create a job's entry in `jobs` -- a job cancelled before it has ever
    # run gets a fresh, never-executed entry with cancelled=True set on it.
    info = state["jobs"].setdefault(
        args.job_id,
        {"run_count": 0, "last_run_tick": None, "last_status": None, "last_exit_code": None},
    )
    info["cancelled"] = True

    write_state_atomic(state_file, state)
    print(f"cancelled: {args.job_id}")
    return EXIT_OK


# --------------------------------------------------------------------------
# Command: history
# --------------------------------------------------------------------------

def cmd_history(args):
    log_dir = args.log_dir
    segments = existing_log_segments(log_dir)
    if not segments:
        print("no history")
        return EXIT_OK

    out = []
    for seg in segments:
        with open(seg, "r", encoding="utf-8") as f:
            seg_lines = f.read().splitlines()
        for raw in reversed(seg_lines):
            if not raw:
                continue
            fields = raw.split("\t")
            if len(fields) != 5:
                continue
            tick, job_id, status, exit_code, reason = fields
            out.append(
                f"tick={tick} job={job_id} status={status} "
                f"exit_code={exit_code} reason={reason}"
            )
            if len(out) >= args.limit:
                break
        if len(out) >= args.limit:
            break

    for line in out:
        print(line)
    return EXIT_OK


# --------------------------------------------------------------------------
# CLI wiring — deliberately not relying on argparse's default exit(2) for
# usage errors, since exit code 2 is reserved for schedule parse errors
# (Section 9).
# --------------------------------------------------------------------------

class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message)


def positive_int(name):
    def _parse(value):
        try:
            n = int(value, 10)
        except ValueError:
            raise UsageError(f"error: --{name} requires a positive integer")
        if n < 1:
            raise UsageError(f"error: --{name} requires a positive integer")
        return n

    return _parse


def build_parser():
    parser = _ArgumentParser(prog="jobsched.py", add_help=True, exit_on_error=False)
    sub = parser.add_subparsers(dest="subcommand")

    p_validate = sub.add_parser("validate", add_help=True, exit_on_error=False)
    p_validate.add_argument("schedule_file")

    p_run = sub.add_parser("run", add_help=True, exit_on_error=False)
    p_run.add_argument("schedule_file")
    p_run.add_argument("--until", default=None)
    p_run.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_run.add_argument("--state-file", dest="state_file", default=None)
    p_run.add_argument("--log-dir", dest="log_dir", default=None)
    p_run.add_argument("--force-recover", action="store_true", dest="force_recover")
    p_run.add_argument("--max-concurrent", dest="max_concurrent", default=None)

    p_status = sub.add_parser("status", add_help=True, exit_on_error=False)
    p_status.add_argument("--state-file", dest="state_file", default=None)

    p_history = sub.add_parser("history", add_help=True, exit_on_error=False)
    p_history.add_argument("--log-dir", dest="log_dir", default=None)
    p_history.add_argument("--limit", dest="limit", default=None)

    p_cancel = sub.add_parser("cancel", add_help=True, exit_on_error=False)
    p_cancel.add_argument("schedule_file")
    p_cancel.add_argument("job_id")
    p_cancel.add_argument("--state-file", dest="state_file", default=None)
    p_cancel.add_argument("--force-recover", action="store_true", dest="force_recover")

    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    try:
        try:
            args = parser.parse_args(argv)
        except argparse.ArgumentError as exc:
            raise UsageError(str(exc))
        except SystemExit as exc:
            # argparse's own -h/--help path, or an internal exit we did not
            # author ourselves via UsageError above.
            raise UsageError(f"argument parsing failed (code {exc.code})")

        if args.subcommand is None:
            raise UsageError("a subcommand is required: validate, run, status, history, cancel")

        if args.subcommand == "validate":
            return cmd_validate(args)

        if args.subcommand == "run":
            if args.until is None:
                raise UsageError("error: --until requires a positive integer")
            args.until = positive_int("until")(args.until)
            args.max_concurrent = (
                positive_int("max-concurrent")(args.max_concurrent)
                if args.max_concurrent is not None
                else None
            )
            args.state_file = args.state_file or default_state_file()
            args.log_dir = args.log_dir or default_log_dir()
            return cmd_run(args)

        if args.subcommand == "status":
            args.state_file = args.state_file or default_state_file()
            return cmd_status(args)

        if args.subcommand == "history":
            args.log_dir = args.log_dir or default_log_dir()
            args.limit = positive_int("limit")(args.limit) if args.limit is not None else 20
            return cmd_history(args)

        if args.subcommand == "cancel":
            args.state_file = args.state_file or default_state_file()
            return cmd_cancel(args)

        raise UsageError(f"unknown subcommand '{args.subcommand}'")

    except UsageError as exc:
        msg = str(exc)
        if not msg.startswith("error:"):
            msg = f"error: {msg}"
        print(msg, file=sys.stderr)
        return EXIT_USAGE
    except ScheduleNotFound as exc:
        print(f"error: schedule file not found: {exc}", file=sys.stderr)
        return EXIT_SCHEDULE_NOT_FOUND
    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_PARSE_ERROR
    except DagError as exc:
        print(f"error: dag: {exc}", file=sys.stderr)
        return EXIT_DAG_ERROR
    except StateCorrupt as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_STATE_CORRUPT
    except StateMismatch:
        print(
            "error: schedule has changed since last run (use --force-recover to reset state)",
            file=sys.stderr,
        )
        return EXIT_STATE_MISMATCH
    except Exception as exc:  # noqa: BLE001 - intentional catch-all, Section 2/9 contract
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
