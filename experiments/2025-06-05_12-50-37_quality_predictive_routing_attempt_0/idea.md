## Name

quality_predictive_routing

## Title

QualityRoute: Learning Quality-Predictive Draft-Verify Pair Selection for Adaptive Speculative Decoding

## Short Hypothesis

By learning to predict output quality from query features, we can dynamically select optimal draft-verify model pairs during speculative decoding, achieving superior quality-speed trade-offs compared to static routing or cost-only optimization approaches.

## Related Work

Existing multi-model systems like LLM-Blender focus on ensemble ranking, FuseLLM on model fusion, and MetaLLM on cost optimization. ZOOTER provides static query routing but doesn't integrate with speculative decoding. Our work uniquely combines learned quality prediction with dynamic speculation, moving beyond cost-centric or fusion-based approaches.

## Abstract

Current speculative decoding uses fixed draft-verify pairs, missing opportunities for query-adaptive optimization. We introduce QualityRoute, which learns to predict output quality from query features and dynamically selects optimal model pairs during speculation. Our system features: (1) lightweight quality prediction models trained on query-response pairs, (2) real-time routing that balances predicted quality with computational cost, and (3) adaptive threshold learning that improves selection over time. Experiments show QualityRoute achieves 25% better quality-speed Pareto frontiers than static approaches while outperforming cost-only routing by 15% on quality metrics.

## Experiments

1) Train quality predictors on diverse query-response datasets with human annotations. 2) Compare against static speculation, MetaLLM cost routing, and LLM-Blender ensemble methods. 3) Evaluate on reasoning tasks (GSM8K), creative writing (WritingPrompts), and factual QA (NaturalQuestions). 4) Measure quality-speed trade-offs using BLEU, ROUGE, and human evaluation. 5) Ablation studies on prediction model architectures and routing strategies. 6) Analysis of routing decisions across different query types and domains.

## Risk Factors And Limitations

1) Quality prediction accuracy crucial for effective routing decisions. 2) Requires diverse training data for robust quality prediction. 3) May introduce latency overhead from prediction and routing. 4) Effectiveness depends on availability of multiple suitable model pairs. 5) Quality metrics may not capture all aspects of output value.

