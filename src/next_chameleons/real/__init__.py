"""Focused modules for the real HF-backed paper and adaptive pipeline.

Split out of the former monolithic ``next_chameleons.real_pipeline`` so each
stage (resolve, data, train, eval, adaptive) is independently readable and
testable. ``next_chameleons.real_pipeline`` remains as a compatibility façade.
"""
