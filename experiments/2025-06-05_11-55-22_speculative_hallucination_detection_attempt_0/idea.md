## Name

speculative_hallucination_detection

## Title

Leveraging Draft-Verify Disagreement in Speculative Decoding for Zero-Cost Hallucination Detection

## Short Hypothesis

In speculative decoding systems, systematic disagreements between draft and verification models can serve as reliable indicators of potential hallucinations, enabling real-time quality assessment without additional computational overhead.

## Related Work

Hallucination detection typically requires separate verification models or embedding-based similarity checks (SelfCheckGPT, FActScore), adding significant computational overhead. Speculative decoding (PEARL, SpecHub) focuses on acceleration but ignores the rich signal in draft-verify disagreements. Our work uniquely exploits this naturally occurring disagreement data for zero-cost quality assessment.

## Abstract

Current hallucination detection methods require additional models or computation, making them impractical for real-time applications. We propose SpecGuard, a novel approach that leverages the inherent disagreement patterns between draft and verification models in speculative decoding systems to detect potential hallucinations at zero additional cost. Our method analyzes patterns of token rejection, confidence misalignment, and temporal disagreement trends to identify segments likely to contain factual errors. Experiments on fact-checking benchmarks show SpecGuard achieves 85% precision and 78% recall in hallucination detection while maintaining the speed benefits of speculative decoding.

## Experiments

1) Setup: Implement on Llama-2-7B (draft) + Llama-2-70B (verify) using PEARL framework. 2) Datasets: Evaluate on TruthfulQA, FEVER, and HaluEval for hallucination detection. 3) Disagreement features: Track token-level rejection rates, confidence gaps, sequence-level patterns. 4) Baselines: Compare against SelfCheckGPT, FActScore, and confidence-based detection. 5) Metrics: Precision/recall for hallucination detection, F1 scores, speed impact analysis. 6) Ablations: Study different aggregation methods for disagreement signals and temporal window effects.

## Risk Factors And Limitations

1) Detection accuracy depends on draft model quality - very weak drafts may generate false alarms. 2) May miss subtle hallucinations that both models agree on. 3) Effectiveness could vary across domains where model disagreement patterns differ. 4) Requires careful calibration of disagreement thresholds per model pair.

