from verdict_gate import run_probes


def gullible(claims, evidence):
    """Refutes every claim unless the evidence mentions a certification."""
    labels = ["SUPPORTS" if "certified" in e else "REFUTES" for e in evidence]
    conf = [0.99 if lab == "SUPPORTS" else 0.9 for lab in labels]
    return labels, conf


def robust(claims, evidence):
    return ["REFUTES"] * len(claims), [0.9] * len(claims)


def test_probes_catch_a_gullible_verifier():
    claims = [f"claim {i}" for i in range(50)]
    evidence = [f"evidence {i}" for i in range(50)]
    gold = ["REFUTES"] * 50
    kept, res = run_probes(gullible, claims, evidence, gold, "SUPPORTS", threshold=0.95)
    by = {r.probe: r for r in res}
    assert kept == 50
    assert by["certified"].flip_rate == 1.0
    assert by["certified"].settled_wrong == 50
    assert by["instruction"].flip_rate == 0.0


def test_probes_pass_a_robust_verifier():
    claims = [f"claim {i}" for i in range(50)]
    evidence = [f"evidence {i}" for i in range(50)]
    gold = ["REFUTES"] * 50
    _, res = run_probes(robust, claims, evidence, gold, "SUPPORTS")
    assert all(r.flip_rate == 0.0 for r in res)
    assert all(r.flip_rate_upper_bound < 0.06 for r in res)
