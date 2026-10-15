"""How many verdicts can a machine settle at a target accuracy, with a guarantee.

The input is a set of verdicts that people have already settled: for each one,
the machine's confidence and whether the machine's verdict agreed with the
human one. The machine settles a verdict when its confidence is at or above a
threshold. For each candidate threshold we bound the error rate among the
verdicts it would settle, and we accept thresholds one at a time from the
strictest to the loosest, stopping at the first one whose bound is too high
(fixed-sequence testing). Because the order is fixed in advance, the chance
that any accepted threshold has a true error rate above the target is at most
delta, with no correction for the number of thresholds tried.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta


def error_upper_bound(errors: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound on an error rate.

    With probability at least 1 - delta, the true error rate is at most this.
    """
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def start_size(eps: float, delta: float) -> int:
    """Fewest settled verdicts at which the sequence starts testing.

    Below this size a threshold would fail even if its true error rate were
    half the allowed one (the bound is too wide), and a failure ends the
    sequence. Starting here keeps small, noisy steps from blocking the rest.
    """
    n = 1
    while error_upper_bound(int(np.floor(n * eps / 2)), n, delta) > eps:
        n += 1
    return n


@dataclass
class Step:
    threshold: float
    settled: int
    errors: int
    coverage: float
    accuracy: float
    accuracy_lower_bound: float
    accepted: bool


@dataclass
class Certificate:
    target_accuracy: float
    delta: float
    n: int
    threshold: float | None
    coverage: float
    settled: int
    errors: int
    accuracy: float | None
    accuracy_lower_bound: float | None
    steps: list[Step] = field(default_factory=list)

    def summary(self) -> str:
        if self.threshold is None:
            return (
                f"No threshold certified: even the most confident verdicts do not reach "
                f"{self.target_accuracy:.0%} accuracy with {1 - self.delta:.0%} confidence "
                f"on {self.n} settled examples."
            )
        return (
            f"Settle verdicts with confidence >= {self.threshold:.4f}: "
            f"{self.coverage:.1%} of verdicts ({self.settled} of {self.n}), "
            f"observed accuracy {self.accuracy:.2%}, "
            f"accuracy >= {self.accuracy_lower_bound:.2%} with {1 - self.delta:.0%} confidence."
        )


def candidate_thresholds(confidence: np.ndarray, max_steps: int = 200) -> np.ndarray:
    """Thresholds from strictest to loosest, at most max_steps of them.

    They are taken at quantiles of the observed confidences so that each step
    admits roughly the same number of extra verdicts.
    """
    qs = np.linspace(1.0, 0.0, max_steps + 1)[1:]
    t = np.unique(np.quantile(confidence, qs, method="higher"))[::-1]
    return t


def certify(
    confidence,
    correct,
    target_accuracy: float = 0.95,
    delta: float = 0.05,
    max_steps: int = 200,
    min_settled: int = 1,
) -> Certificate:
    """Largest share of verdicts the machine can settle at the target accuracy.

    confidence: machine confidence per verdict (higher = more sure).
    correct: 1 if the machine verdict matched the human verdict, else 0.
    """
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    if c.shape != y.shape or c.ndim != 1:
        raise ValueError("confidence and correct must be 1-D arrays of the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("correct must contain only 0 and 1")
    n = len(c)
    eps = 1.0 - target_accuracy
    # Thresholds that settle too few verdicts are skipped (see start_size).
    # Which ones are skipped depends on the confidences alone, not on which
    # verdicts were correct, so the order of tests is fixed before the
    # correctness labels are looked at.
    floor = max(min_settled, start_size(eps, delta))
    steps: list[Step] = []
    best: Step | None = None
    for t in candidate_thresholds(c, max_steps):
        mask = c >= t
        m = int(mask.sum())
        if m < floor:
            continue
        k = int(m - y[mask].sum())
        ub = error_upper_bound(k, m, delta)
        ok = ub <= eps
        step = Step(
            threshold=float(t),
            settled=m,
            errors=k,
            coverage=m / n,
            accuracy=1 - k / m,
            accuracy_lower_bound=1 - ub,
            accepted=ok,
        )
        steps.append(step)
        if not ok:
            break
        best = step
    if best is None:
        return Certificate(target_accuracy, delta, n, None, 0.0, 0, 0, None, None, steps)
    return Certificate(
        target_accuracy,
        delta,
        n,
        best.threshold,
        best.coverage,
        best.settled,
        best.errors,
        best.accuracy,
        best.accuracy_lower_bound,
        steps,
    )


def certify_by_group(
    confidence,
    correct,
    group,
    target_accuracy: float = 0.95,
    delta: float = 0.05,
    max_steps: int = 200,
) -> dict[str, Certificate]:
    """One certificate per group (for example, per product category).

    delta is split evenly across groups (Bonferroni), so all the group
    certificates hold together with probability at least 1 - delta.
    """
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    g = np.asarray(group).astype(str)
    names = sorted(set(g.tolist()))
    d = delta / len(names)
    return {
        name: certify(c[g == name], y[g == name], target_accuracy, d, max_steps)
        for name in names
    }


def audit_bound(errors: int, audited: int, delta: float = 0.05) -> tuple[float, float]:
    """Accuracy of machine-settled verdicts in production, from a random audit.

    Audit a uniformly random sample of the verdicts the machine settled; this
    returns (observed accuracy, lower bound holding with probability 1 - delta).
    The bound does not depend on why the machine is wrong, so it also covers
    failure modes nobody anticipated.
    """
    if audited == 0:
        return float("nan"), 0.0
    return 1 - errors / audited, 1 - error_upper_bound(errors, audited, delta)


def audit_alarm(
    errors: int, audited: int, target_accuracy: float = 0.95, delta: float = 0.05
) -> tuple[bool, float]:
    """Single-look test: is accuracy below the target, given one random audit?

    Use it once, on a finished audit. To watch a stream of audits, use
    AuditMonitor instead. Returns (alarm, accuracy upper bound). The alarm is raised only when even
    the upper bound (probability 1 - delta) is below the target, so a false
    alarm on a verifier that meets the target has probability at most delta.
    """
    if audited == 0 or errors == 0:
        return False, 1.0
    err_lower = float(beta.ppf(delta, errors, audited - errors + 1))
    upper = 1 - err_lower
    return upper < target_accuracy, upper


class AuditMonitor:
    """Continuous audit of settled verdicts, valid however often you look.

    audit_alarm is a single-look test: checking it after every audit inflates
    false alarms (at exactly the target accuracy, checking after each of 2,500
    audits raises a false alarm about a third of the time). This monitor bets
    against "error rate <= 1 - target" and raises the alarm when its wealth
    reaches 1 / delta. Under that hypothesis the wealth is a nonnegative
    supermartingale, so by Ville's inequality the chance of a false alarm at
    any point, ever, is at most delta. The wealth averages several betting
    fractions so it adapts to how far below target the verifier has fallen.
    """

    def __init__(self, target_accuracy: float = 0.95, delta: float = 0.05):
        self.eps = 1.0 - target_accuracy
        self.delta = delta
        cap = 1.0 / self.eps
        self.lambdas = np.array([cap * f for f in (0.02, 0.05, 0.1, 0.2, 0.4, 0.6)])
        self.log_w = np.zeros_like(self.lambdas)
        self.audited = 0
        self.errors = 0

    def update(self, correct: bool) -> bool:
        """Record one audited verdict; return True once the alarm is raised."""
        x = 0.0 if correct else 1.0
        self.audited += 1
        self.errors += int(not correct)
        self.log_w += np.log1p(self.lambdas * (x - self.eps))
        return self.alarm

    @property
    def wealth(self) -> float:
        m = self.log_w.max()
        return float(np.exp(m) * np.mean(np.exp(self.log_w - m)))

    @property
    def alarm(self) -> bool:
        return self.wealth >= 1.0 / self.delta


def audits_needed(
    target_accuracy: float, expected_accuracy: float, delta: float = 0.05, max_n: int = 1_000_000
) -> int | None:
    """Smallest audit size that would confirm the target if accuracy is as expected.

    Assumes the audit sees errors at the expected rate. Returns None when the
    expected accuracy does not exceed the target (no audit size can confirm it).
    """
    if expected_accuracy <= target_accuracy:
        return None
    eps = 1 - target_accuracy
    e = 1 - expected_accuracy
    lo, hi = 1, 1
    while hi <= max_n and error_upper_bound(int(round(e * hi)), hi, delta) > eps:
        lo, hi = hi, hi * 2
    if hi > max_n:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if error_upper_bound(int(round(e * mid)), mid, delta) <= eps:
            hi = mid
        else:
            lo = mid + 1
    return lo
