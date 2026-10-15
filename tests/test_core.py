import numpy as np
import pytest

from verdict_gate import AuditMonitor, audit_alarm, audit_bound, audits_needed, certify, certify_by_group, error_upper_bound
from verdict_gate.cli import main


def test_bound_zero_errors_matches_closed_form():
    # With no errors in n, the Clopper-Pearson upper bound is 1 - delta**(1/n).
    for n in (1, 10, 59, 300):
        assert error_upper_bound(0, n, 0.05) == pytest.approx(1 - 0.05 ** (1 / n))


def test_bound_edge_cases():
    assert error_upper_bound(0, 0, 0.05) == 1.0
    assert error_upper_bound(5, 5, 0.05) == 1.0
    assert error_upper_bound(3, 100, 0.05) > 0.03


def test_perfect_verdicts_certify_everything():
    rng = np.random.default_rng(0)
    conf = rng.random(2000)
    cert = certify(conf, np.ones(2000, dtype=int), 0.95, 0.05)
    assert cert.coverage == 1.0
    assert cert.accuracy_lower_bound >= 0.95


def test_useless_verdicts_certify_nothing():
    rng = np.random.default_rng(1)
    conf = rng.random(2000)
    correct = rng.integers(0, 2, 2000)
    cert = certify(conf, correct, 0.95, 0.05)
    assert cert.threshold is None
    assert cert.coverage == 0.0


def _true_error_above(t):
    # Confidence c ~ Uniform(0.5, 1) and P(wrong | c) = (1 - c) / 4, so the error
    # rate among verdicts with c >= t is the mean of (1 - c) / 4 over [t, 1]:
    # (1 - t) / 8. It equals 5% at t = 0.6, i.e. 80% of verdicts.
    return (1 - t) / 8


def test_guarantee_holds_in_simulation():
    """The certified threshold should exceed the target error at most delta of the time."""
    rng = np.random.default_rng(2)
    target, delta, trials, n = 0.95, 0.1, 400, 1500
    failures = 0
    certified = 0
    for _ in range(trials):
        conf = rng.uniform(0.5, 1.0, n)
        correct = (rng.random(n) >= (1 - conf) / 4).astype(int)
        cert = certify(conf, correct, target, delta, max_steps=100)
        if cert.threshold is None:
            continue
        certified += 1
        if _true_error_above(cert.threshold) > 1 - target:
            failures += 1
    assert certified > trials * 0.9
    # delta = 0.1; allow sampling noise over 400 trials.
    assert failures / trials <= delta + 0.04


def test_certify_rejects_bad_input():
    with pytest.raises(ValueError):
        certify([0.9, 0.8], [1])
    with pytest.raises(ValueError):
        certify([0.9, 0.8], [1, 2])


def test_by_group_splits_delta():
    rng = np.random.default_rng(3)
    conf = rng.random(3000)
    correct = np.ones(3000, dtype=int)
    group = np.array(["a", "b", "c"] * 1000)
    certs = certify_by_group(conf, correct, group, 0.95, 0.06)
    assert set(certs) == {"a", "b", "c"}
    assert all(c.delta == pytest.approx(0.02) for c in certs.values())


def test_audit_bound_and_plan():
    acc, lb = audit_bound(2, 400, 0.05)
    assert acc == pytest.approx(0.995)
    assert 0.98 < lb < acc
    n = audits_needed(0.95, 0.99, 0.05)
    assert n is not None
    assert 1 - error_upper_bound(round(0.01 * n), n, 0.05) >= 0.95
    assert audits_needed(0.95, 0.94) is None


def test_cli_certify(tmp_path, capsys):
    rng = np.random.default_rng(5)
    p = tmp_path / "v.csv"
    rows = ["confidence,machine_verdict,human_verdict,category"]
    for i in range(3000):
        c = rng.uniform(0.5, 1.0)
        ok = rng.random() >= (1 - c) / 4
        rows.append(f"{c},{'A' if ok else 'B'},A,{'x' if i % 2 else 'y'}")
    p.write_text("\n".join(rows), encoding="utf-8")
    assert main(["certify", str(p)]) == 0
    assert "Settle verdicts" in capsys.readouterr().out
    assert main(["certify", str(p), "--by", "category"]) == 0
    assert "All groups" in capsys.readouterr().out


def test_audit_alarm():
    # 30 errors in 200 audited: accuracy about 85%, clearly below 95%.
    alarm, upper = audit_alarm(30, 200, 0.95, 0.05)
    assert alarm and upper < 0.95
    # 4 errors in 200: 98% observed, no alarm.
    assert audit_alarm(4, 200, 0.95, 0.05)[0] is False
    assert audit_alarm(0, 50)[0] is False


def test_audit_alarm_false_alarm_rate():
    rng = np.random.default_rng(6)
    alarms = sum(audit_alarm(int(rng.binomial(300, 0.05)), 300, 0.95, 0.05)[0] for _ in range(2000))
    assert alarms / 2000 <= 0.05 + 0.015


def _first_alarm(err_rate, n_audits, rng):
    mon = AuditMonitor(0.95, 0.05)
    for i in range(n_audits):
        if mon.update(rng.random() >= err_rate):
            return i + 1
    return None


def test_monitor_false_alarms_stay_below_delta_when_watched_continuously():
    rng = np.random.default_rng(7)
    runs = 400
    fa = sum(_first_alarm(0.05, 2500, rng) is not None for _ in range(runs))
    assert fa / runs <= 0.05 + 0.02


def test_monitor_catches_a_drop():
    rng = np.random.default_rng(8)
    hits = [_first_alarm(0.10, 5000, rng) for _ in range(200)]
    assert all(h is not None for h in hits)
    assert np.median(hits) < 1000
