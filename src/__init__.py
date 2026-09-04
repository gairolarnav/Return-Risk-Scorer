"""
Return-risk scorer — the importable package behind every number in this repo.

Notebooks and `scripts/` import from here and hold no logic of their own. Two
divergent implementations of a feature builder is the failure mode that
invalidates every result a notebook produces (docs/ARCHITECTURE.md §8), so a
transform lives in exactly one module and everything else calls it.

Module map:
    data_gate         data gate: customer viability, split strategy, leakage sweep
    features          feature engineering + FULL/TESTBED track definitions
    ablation          degeneracy ladder (diagnostic evidence, not feature selection)
    model             trains either track, saves model + proba + metrics JSON
    evaluate          per-class metrics, PR curves, cost matrix, both sweep axes
    infer             single-record + batch scoring -> recommended intervention
    explain           per-class SHAP on both tracks
    smote_experiment  SMOTE vs class-weighted on testbed
    segment_audit     segment-level FPR by order-value bucket
"""
