"""verdict-gate: how many verdicts a machine can settle at a target accuracy, with a guarantee."""

from .core import (
    AuditMonitor,
    Certificate,
    audit_alarm,
    audit_bound,
    audits_needed,
    certify,
    certify_by_group,
    error_upper_bound,
)
from .probe import PROBES, run_probes

__all__ = [
    "AuditMonitor",
    "Certificate",
    "PROBES",
    "audit_alarm",
    "audit_bound",
    "audits_needed",
    "certify",
    "certify_by_group",
    "error_upper_bound",
    "run_probes",
]
__version__ = "0.1.0"
