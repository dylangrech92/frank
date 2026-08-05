## Diagnosis

This bug has two independent, cooperating causes.

First, `queueworks/retry.py`'s `RetryPolicy.should_retry()` compares with
`job.attempts < self.max_retries`, which is off by one against the
documented semantics (retry while attempts is 1, 2, or 3, give up at 4) --
it makes the job give up after only 3 total attempts instead of 4.

Second, even once that comparison is corrected to `<=` so the job reaches
its 4th attempt, `queueworks/worker.py`'s `_run_with_retries()` applies the
`max_delay` cap backwards: `if raw_delay < self.retry_policy.max_delay:
delay = self.retry_policy.max_delay else: delay = raw_delay`. That clamps
*small* raw delays up to `max_delay` (wasting time on early retries) and
lets *large* raw delays through completely uncapped on later retries -- the
opposite of what a cap should do; the condition is essentially backwards.
Fixing only `should_retry` does make the job reach 4 attempts, but the
delay computed for the 3rd retry (`base_delay * backoff_factor ** attempts`
= 0.05 * 4**3 = 3.2s) sails straight past `max_delay` because the clamp
condition is inverted.

## Fix

Change `should_retry` to `job.attempts <= self.max_retries`, and change the
worker's cap logic to `delay = min(raw_delay, self.retry_policy.max_delay)`
so every delay -- early or late -- is correctly capped at `max_delay`.
