"""Retry policy: how many times to retry a job, and how long to wait.

The two concerns are kept separate on purpose. :meth:`RetryPolicy.should_retry`
answers "is there another attempt left", and :meth:`RetryPolicy.compute_backoff`
answers "how long should the next attempt wait" -- it returns the *raw*,
uncapped exponential value. Applying the ceiling (``max_delay``) is left to
the caller (see :mod:`queueworks.worker`), the same way ``time.sleep``
callers are responsible for clamping their own inputs.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    """Configuration for how a failing job should be retried.

    ``max_retries`` is the number of retries allowed *after* the initial
    attempt -- with the default of 3, a job that keeps failing is attempted
    4 times in total before being marked permanently failed.
    """

    max_retries: int = 3
    base_delay: float = 0.05
    backoff_factor: float = 4.0
    max_delay: float = 1.0

    def should_retry(self, job) -> bool:
        """Return True if ``job`` has another retry attempt available.

        ``job.attempts`` counts attempts already made (including the
        initial one), so with ``max_retries=3`` a job should still be
        retried after its 1st, 2nd, and 3rd attempts -- i.e. while
        ``job.attempts`` is 1, 2, or 3 -- and give up once it reaches 4.
        """

        return job.attempts <= self.max_retries

    def compute_backoff(self, attempts: int) -> float:
        """Return the raw (uncapped) backoff delay for ``attempts``.

        ``attempts`` is the 1-indexed attempt number that just failed.
        Callers must clamp the result to ``max_delay`` themselves.
        """

        return self.base_delay * (self.backoff_factor ** attempts)
