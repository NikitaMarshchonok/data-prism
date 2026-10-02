"""Fail-closed deployment declarations for runtime state durability.

The application cannot prove that an arbitrary path is backed by durable
infrastructure.  This contract therefore records an explicit operator
declaration and prevents a production profile from reporting ready when that
declaration is absent or internally inconsistent.
"""

from __future__ import annotations

from typing import Any


RUNTIME_STORAGE_CONTRACT = "runtime-storage-v1"
DEPLOYMENT_PROFILES = frozenset({"development", "demo", "production"})
STATE_DURABILITY_CLASSES = frozenset({"ephemeral", "persistent"})


def _choice(value: object, allowed: frozenset[str]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in allowed else None


def evaluate_runtime_storage_contract(
    *,
    deployment_profile: object,
    state_durability: object,
    state_directory_configured: bool,
) -> dict[str, Any]:
    """Return a bounded readiness contract without claiming mount verification."""

    profile = _choice(deployment_profile, DEPLOYMENT_PROFILES)
    durability = _choice(state_durability, STATE_DURABILITY_CLASSES)
    explicit_state_directory = state_directory_configured is True
    issues: list[str] = []
    warnings: list[str] = []

    if profile is None:
        issues.append(
            "DATA_PRISM_DEPLOYMENT_PROFILE must be development, demo, or production."
        )
    if durability is None:
        issues.append(
            "DATA_PRISM_STATE_DURABILITY must be ephemeral or persistent."
        )

    if durability == "persistent" and not explicit_state_directory:
        issues.append(
            "Persistent state requires an explicit DATA_PRISM_STATE_DIR."
        )

    if profile == "production":
        if durability != "persistent":
            issues.append(
                "Production deployment requires DATA_PRISM_STATE_DURABILITY=persistent."
            )
        if not explicit_state_directory:
            issues.append(
                "Production deployment requires an explicit DATA_PRISM_STATE_DIR."
            )

    if durability == "ephemeral":
        warnings.append(
            "Runtime state can be lost during restart, redeploy, or host replacement."
        )
    elif durability == "persistent":
        warnings.append(
            "Persistence is operator-declared; verify the backing store and recovery procedure."
        )

    production_state_required = profile == "production"
    production_state_satisfied = bool(
        production_state_required
        and durability == "persistent"
        and explicit_state_directory
        and profile is not None
        and not issues
    )

    return {
        "contract": RUNTIME_STORAGE_CONTRACT,
        "deployment_profile": profile or "invalid",
        "state_durability": durability or "invalid",
        "state_directory_configured": explicit_state_directory,
        "production_state_required": production_state_required,
        "production_state_satisfied": production_state_satisfied,
        "assurance": "operator-declared",
        "issues": issues,
        "warnings": warnings,
    }
