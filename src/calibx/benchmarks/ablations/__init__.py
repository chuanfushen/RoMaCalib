"""Paper ablation protocols, including masking and matcher comparisons."""

from .external_matchers import ExternalMatcherSettings, create_external_matcher
from .matchers import MatcherAblationCondition, assert_matcher_comparable
from .sam import SamAblationCondition, SamAblationPair
from .sam_executor import (
    SamTable4Execution,
    SamTable4Hooks,
    configure_sam_table4_determinism,
    condition_plan,
    execute_sam_table4_condition,
    execute_sam_table4_pair,
)

__all__ = [
    "MatcherAblationCondition",
    "ExternalMatcherSettings",
    "SamAblationCondition",
    "SamAblationPair",
    "SamTable4Execution",
    "SamTable4Hooks",
    "assert_matcher_comparable",
    "configure_sam_table4_determinism",
    "condition_plan",
    "create_external_matcher",
    "execute_sam_table4_condition",
    "execute_sam_table4_pair",
]
