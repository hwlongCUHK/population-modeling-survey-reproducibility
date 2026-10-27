# Executed supplementary protocol

## Fixed-post semantic aggregation

With per-post predictions held fixed, weighting group means by their sizes equals the overall post mean. The maximum archived numerical discrepancy is approximately `1.67e-16`. Selection, weights and joint-context prediction are consequently documented separately.

## Matched evidence selection

- Semantic mapping: 500 posts per condition; normalized BGE-base-zh-v1.5 embeddings; anchor format in `prompts/semantic_anchor_template.txt`; temperature 0.1.
- Direct retrieval ranks posts by cosine similarity to question text plus ordered answer options.
- The direct-clustered comparison uses the same candidate corpus as the frozen cluster-selection arm. The direct-full corpus comparison is also released separately.
- The cached top-ten cluster choice is shared across questions. It is distinct from the question-specific ranking used for direct individual-post retrieval.
- LLM selection control: 80 posts per condition, split into ten groups of eight with the same permutation seed 101. Selectors are compared within each estimator, not at a common 500-post budget across estimators.

## Real versus random groups

- Same 80 posts, ten groups of eight, per-post character window, prompts, decoding and arithmetic aggregation.
- Five random partitions use seeds 101, 202, 303, 404 and 505.
- The random comparator is the mean of the five JS scores **within each question**, followed by question-level comparison. It is not JS of an averaged prediction vector.
- One random group returns an all-zero vector. That question has no complete five-partition comparator, so the primary paired comparison uses 12 questions. All records and the failure remain visible.
- Four questions improve and eight worsen with real grouping; the original-six subset is reported separately.

## Capacity-constrained K-means

- Fixed selected evidence: 80 posts, 10 groups, capacity 8.
- Cached BGE-base-zh-v1.5 full-post embeddings, default encoder truncation, L2 normalization, dimension 768, no UMAP.
- K-means++ initialization; optimal Hungarian slot assignment; centroid means; Euclidean SSE.
- Seed 42, ten initializations, maximum 100 iterations; select the initialization with minimum SSE without using survey targets.
- Run real and K-means arms together: 13 questions × 10 groups × 2 arms = 260 requests.
- All ten initializations converge. The selected solution has SSE 25.0493 versus 24.1714 for the real grouping; it is a local solution rather than a certified global minimum.
- Mean within-group cosine similarity: real 0.6547; K-means 0.6422. Adjusted Rand index between the partitions: 0.3926.

## LLM generation and parsing

The recorded model is Qwen3-8B. Each post contributes its first 100 Unicode characters. No group/topic names are supplied. Generation uses bf16, SDPA, batch size four, greedy decoding (`do_sample=False`), thinking disabled and at most 160 new tokens. The executed server used an A100 with 80 GB memory, with a per-process memory cap of 0.30.

An output must parse as a numeric array or an object whose only key is `distribution`, with the expected option count, finite nonnegative values and positive sum. Valid vectors are normalized by their sum. All-zero, malformed or wrong-dimensional outputs are invalid. A question-arm is averaged only when all ten groups are valid. There is no fallback distribution, averaging over surviving groups or selection of a preferred retry.

## Statistics

- JS divergence uses natural logarithms and is not square-root JS distance.
- Differences use the exact complete-pair question subset, with direction stated for each comparison.
- Raw p values enumerate all sign flips of paired question differences and use a two-sided absolute-mean statistic.
- The original three-comparison Holm family contains semantic direct-clustered/cluster-selected, semantic direct-full/cluster-selected, and LLM real/mean-random comparisons.
- The additional LLM selection and K-means comparisons retain raw p values and are not silently included in that original correction family.
- Descriptive intervals use 20,000 bootstrap resamples of questions and seed 20261026. Questions share a domain and source evidence; the intervals describe the evaluated question set, rather than independently sampled populations or domains.
- The original-six subset is fixed in `results/question_index.json` and reported as a secondary description.

Survey target frequencies enter evaluation after model predictions are produced. The public release contains derived error scores; the target distributions remain separately held.
