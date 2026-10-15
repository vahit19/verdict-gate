"""Command line: verdict-gate certify | audit | plan."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from .core import audit_bound, audits_needed, certify, certify_by_group


def _read(path: str, conf_col: str, correct_col: str | None, machine_col: str | None,
          human_col: str | None, group_col: str | None):
    conf, correct, group = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conf.append(float(row[conf_col]))
            if correct_col:
                correct.append(int(float(row[correct_col])))
            else:
                correct.append(int(row[machine_col].strip() == row[human_col].strip()))
            if group_col:
                group.append(row[group_col])
    return conf, correct, (group or None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verdict-gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("certify", help="share of verdicts the machine can settle at a target accuracy")
    c.add_argument("csv")
    c.add_argument("--target", type=float, default=0.95, help="target accuracy (default 0.95)")
    c.add_argument("--delta", type=float, default=0.05, help="allowed failure probability (default 0.05)")
    c.add_argument("--confidence", default="confidence", help="confidence column")
    c.add_argument("--correct", default=None, help="0/1 column: machine verdict matched human verdict")
    c.add_argument("--machine", default="machine_verdict")
    c.add_argument("--human", default="human_verdict")
    c.add_argument("--by", default=None, help="group column, e.g. category")
    c.add_argument("--json", action="store_true")

    a = sub.add_parser("audit", help="accuracy bound from a random audit of settled verdicts")
    a.add_argument("--errors", type=int, required=True)
    a.add_argument("--audited", type=int, required=True)
    a.add_argument("--delta", type=float, default=0.05)

    pl = sub.add_parser("plan", help="audit size needed to confirm a target accuracy")
    pl.add_argument("--target", type=float, default=0.95)
    pl.add_argument("--expected", type=float, required=True, help="accuracy you expect, e.g. 0.98")
    pl.add_argument("--delta", type=float, default=0.05)

    args = p.parse_args(argv)
    if args.cmd == "certify":
        conf, correct, group = _read(args.csv, args.confidence, args.correct, args.machine,
                                     args.human, args.by)
        if args.by:
            certs = certify_by_group(conf, correct, group, args.target, args.delta)
            if args.json:
                print(json.dumps({k: _cert_json(v) for k, v in certs.items()}, indent=2))
            else:
                total = sum(v.n for v in certs.values())
                settled = sum(v.settled for v in certs.values())
                for k, v in certs.items():
                    print(f"{k}: {v.summary()}")
                print(f"\nAll groups: {settled} of {total} verdicts settled ({settled / total:.1%}).")
        else:
            cert = certify(conf, correct, args.target, args.delta)
            print(json.dumps(_cert_json(cert), indent=2) if args.json else cert.summary())
    elif args.cmd == "audit":
        acc, lb = audit_bound(args.errors, args.audited, args.delta)
        print(f"Observed accuracy {acc:.2%}; accuracy >= {lb:.2%} with {1 - args.delta:.0%} confidence.")
    else:
        n = audits_needed(args.target, args.expected, args.delta)
        if n is None:
            print("Expected accuracy does not exceed the target; no audit size can confirm it.")
        else:
            print(f"Audit {n} randomly chosen settled verdicts.")
    return 0


def _cert_json(c):
    d = dict(c.__dict__)
    d["steps"] = [s.__dict__ for s in c.steps]
    return d


if __name__ == "__main__":
    sys.exit(main())
