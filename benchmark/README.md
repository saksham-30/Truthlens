# TruthLens benchmark audit

This directory is separate from the raw dataset and existing processed splits. It stores a reproducible, grouped benchmark built to limit leakage from the synthetic generator.

## Provenance
The raw CSV contains 1000 rows but only 150 unique texts. The same templates are repeated across a narrow topic list, so the benchmark is synthetic and template-driven rather than true source-based.

## Split method
Because no source IDs or template IDs are available, the benchmark groups by a deterministic topic-masked template signature and then applies GroupShuffleSplit with seed 42. The resulting split is group-disjoint, but it is not class-balanced: validation contains only AI-labeled examples and test contains only human-labeled examples. The saved model metrics were calculated across all rows, not on the test split, so they are not held-out test performance.

## Recommendation
This benchmark is not currently suitable for a two-class held-out evaluation or a jury demonstration. The generator creates repeated templates and topic patterns, and the current test partition contains only one class. Use it for dataset and split diagnostics only; do not interpret the saved metrics as test performance.
