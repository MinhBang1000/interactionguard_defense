"""Shared training eligibility; test scenario identity never enters model text."""
from research_runner.implementation.data.data_preprocess import VERSION

POLICY_VERSION = "exclude_correlated_and_unresolved_v1"


def select_training_rows(rows, *, benign_only=False):
    """Exclude correlated adaptation and unresolved annotations, not all attack attempts.

    A verified negative prefix may originate in an attack-attempt episode.
    This is an eligibility filter, not an independent semantic label validator.
    """
    if any(row.get("preprocessing_version") != VERSION for row in rows):
        raise ValueError("Regenerate prefixes using the current data preparation workflow first")
    return [row for row in rows
            if row.get("attack_type") != "correlated"
            and not row.get("annotation_requires_review", True)
            and (not benign_only or row["label"] == 0)]
