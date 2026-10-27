# Population Modeling in LLM-Based Survey Simulation

Code and derived results accompanying **Population Modeling Should Be a First-Order Design Problem in LLM-Based Survey Simulation**.

This release covers the executed supplementary controls: matched evidence selection, fixed-post semantic aggregation, real versus random LLM grouping, and a capacity-constrained K-means grouping control. It makes the choices of evidence selection, aggregation and joint-context estimation inspectable.

**Original posts and survey data are excluded.** The repository includes dataset-free prompt templates, newly assigned question/post identifiers, fixed selections, group assignments, numeric model responses, per-question error scores, seeds, decoding settings and public-file hashes. No source-text mapping is released.

## Reproduce the published statistics

Requires Python 3.10+ and NumPy; no GPU, model download or API key is needed.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/reproduce_statistics.py
python -m unittest discover -s tests -v
```

To save the recomputed statistics:

```bash
python scripts/reproduce_statistics.py --out out/recomputed_statistics.json
```

The script reconstructs every released paired comparison, descriptive bootstrap interval and the original three-comparison Holm adjustment from the anonymous per-question JS scores. It also re-parses all 1,300 numeric model responses, including the single invalid response, using the executed parsing rules.

Recomputing each JS score from model predictions requires the separately held survey target distributions. Regenerating prompts, embeddings or model predictions requires separately authorized source inputs. The interfaces are documented in [Private input and execution guide](docs/PRIVATE_EXECUTION.md).

## Main observed results

JS divergence uses natural logarithms; smaller is better. Differences and p values are paired within the indicated question set.

| Comparison A / B | Complete pairs | Mean JS A | Mean JS B | A minus B | Raw two-sided p |
|---|---:|---:|---:|---:|---:|
| Semantic direct retrieval / cluster selection | 13 | 0.029548 | 0.032470 | -0.002922 | 0.322754 |
| LLM direct retrieval / cluster selection, both random partition 101 | 13 | 0.083037 | 0.062016 | +0.021021 | 0.158691 |
| LLM real groups / mean score of five random partitions | 12 | 0.069266 | 0.074588 | -0.005322 | 0.549316 |
| LLM capacity-constrained K-means / rerun real groups | 13 | 0.058819 | 0.064329 | -0.005510 | 0.278320 |

The selection rankings differ between estimators. In the K-means comparison, error decreases on 9 of 13 questions, while mean within-group cosine similarity decreases from 0.655 to 0.642. These observations motivate evaluating evidence organization against the survey task alongside semantic coherence. Per-question scores expose effects hidden by an aggregate mean.

The K-means-minus-real descriptive 95% interval is `[-0.014566, +0.002391]`. Its largest gains occur on two questions. The interval resamples the 13 questions within the same domain and shared evidence pool. The raw p values describe the executed comparisons; they do not establish an equivalence result.

## Experiments and files

| Location | Content |
|---|---|
| `src/population_survey/` | Natural-log JS, paired statistics, parsing/prompt functions, capacity-constrained K-means |
| `prompts/` | Prompt templates with placeholders, containing no survey questions, answer options or post snippets |
| `results/question_index.json` | Anonymous question identifiers, option counts and original-six membership |
| `results/matched_controls/` | Frozen selections/grouping, numeric outputs, semantic predictions, per-question scores, reference statistics and protocol |
| `results/kmeans/` | Frozen group assignments, numeric outputs, per-question scores, grouping diagnostics and protocol |
| `scripts/reproduce_statistics.py` | Offline reconstruction of public statistical summaries |
| `scripts/run_private_*.py` | Model execution using external private input directories |
| `scripts/evaluate_private_*.py` | Prediction-to-JS evaluation using externally supplied targets |
| `tests/` | Numerical, grouping, public-data and reproducibility checks; synthetic fixtures are test-only |
| `RELEASE_MANIFEST.json` | SHA-256 hashes of published files, excluding the manifest itself |

The matched controls contain 1,040 responses: 1,039 valid and one all-zero probability vector. A failed group invalidates its question-arm, yielding 12 complete real/random pairs. The K-means run contains 260 valid responses and 13 complete pairs. Its 130 real-arm responses are byte-identical to the earlier real arm.

## Protocol and data access

See [Executed protocol](docs/PROTOCOL.md), [Data and access](docs/DATA_ACCESS.md), and [Private execution](docs/PRIVATE_EXECUTION.md).

The K-means control uses cached **BGE-base-zh-v1.5**, full-post, normalized 768-dimensional features without UMAP, and Qwen3-8B with the frozen prompt. It regroups the same 80 selected posts into ten groups of eight. The original manuscript's BGE-small/UMAP/HDBSCAN pipeline is a separate grouping construction; this release records the actual supplemental settings.

The human evidence audit has no completed results in this release. No synthetic labels are presented as human annotations.

## Verification

The public statistical reconstruction and eight offline test cases passed before publication, including a synthetic end-to-end check of private K-means preparation and evaluation. Optional CUDA model runners retain the executed generation settings and have been adapted to accept local model paths and external private directories. GPU inference wrappers have been syntax-checked; the completed numeric results are the archived experimental outputs, rather than a new model run performed during publication.
