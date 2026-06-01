"""Named benchmark suites used by reports and docs."""

CAPABILITY_TASKS: tuple[str, ...] = (
    "mmlu",
    "gsm8k",
    "boolq",
    "crows_pairs",
    "social_iqa",
    "truthfulqa_mc2",
    "piqa",
)

SAFETY_LIKELIHOOD_DOMAINS: tuple[str, ...] = (
    "dolus_deception",
    "apollo_repe_deception",
    "circuit_breakers_harmful",
    "synthetic_harmful",
)

PAPER_REPLICATION_CHECKS: tuple[str, ...] = (
    "all_11_benign_concepts",
    "table_1_trigger_scenarios",
    "single_layer_12_linear_probe",
    "mlp_hidden_64_probe",
    "attention_4_head_probe",
    "four_and_eight_layer_ensembles",
    "fixed_1pct_fpr_thresholds",
    "three_seed_bootstrap_reports",
    "base_trigger_control",
    "capability_tax_lm_eval",
    "safety_likelihood_retention",
    "language_generalization_appendix",
)
