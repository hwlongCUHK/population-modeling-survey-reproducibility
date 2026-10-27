# Execution with separately authorized inputs

The public statistical workflow needs only NumPy. The optional model workflow additionally needs local BGE and Qwen model directories, the dependencies in `requirements-model.txt`, and a suitable CUDA environment.

The completed runs used Python 3.10.16, torch 2.5.1 and transformers 4.51.3. The cached-feature K-means run recorded sentence-transformers 5.4.0 in its surrounding environment but did not load that library for feature preparation. Prefer the existing verified server environment. Requirements list compatible ranges for optional setup; they are not a new GPU-tested lockfile.

## Private input contract

Use a directory **outside this repository**, such as a separately managed private workspace. The scripts reject run/output directories inside the repository.

For the matched controls, supply `inputs.json` with:

```json
{
  "questions": [
    {
      "key": "q01",
      "question": "<AUTHORIZED PRIVATE QUESTION TEXT>",
      "options": ["<AUTHORIZED OPTION 1>", "<AUTHORIZED OPTION 2>"],
      "cluster_weights": {"0": 1.0}
    }
  ],
  "posts": [
    {
      "id": "p00001",
      "corpus": "clustered",
      "text": "<AUTHORIZED PRIVATE POST TEXT>",
      "cluster": 0
    }
  ],
  "source_hashes": {}
}
```

This is a **schema illustration**, not a runnable empirical or synthetic study dataset. The actual run needs all 13 questions, original option ordering, full/clustered corpus rows in their original order, cluster labels and cached cluster weights. Post IDs must match the separately supplied mapping to the public anonymous IDs. The public ID map cannot recover the source text.

Provide `targets.json` separately, mapping `q01` through `q13` to nonnegative answer counts or probabilities in the same option order. These are evaluation-only inputs.

For K-means preparation, supply a separate `matched_input.json` with `questions`, the 80 selected `posts`, and `real_groups` (ten ordered lists of eight IDs). Question entries need `key`, `question`, and `options`; post entries need `id` and `text`.

## Matched-control run

Place the authorized `inputs.json` in your external run directory. Model paths below are placeholders to be replaced with actual local paths.

```bash
python scripts/run_private_matched_controls.py \
  --run-dir /path/to/private/matched_run \
  --model /path/to/local/Qwen3-8B \
  --encoder /path/to/local/bge-base-zh-v1.5
```

This runs 1,040 group prompts and saves private prompt instances and derived predictions in that external directory. Embedding caches must correspond to the exact input row order and encoder; use a fresh run directory when inputs change.

After prediction is complete, place the authorized `targets.json` in the same private directory, then run:

```bash
python scripts/evaluate_private_matched_controls.py \
  --run-dir /path/to/private/matched_run
```

## Capacity-constrained K-means run

For the executed representation, supply an authorized NPZ containing `ids` (Unicode array in exactly the order of `matched_input.json` posts) and `vectors` (80×768 normalized BGE-base features). NPZ loading disables pickle.

```bash
python scripts/prepare_private_kmeans.py \
  --input /path/to/private/matched_input.json \
  --features-npz /path/to/private/bge_base80.npz \
  --feature-description 'Cached full-post BGE-base-zh-v1.5, L2 normalization, no UMAP' \
  --out /path/to/private/kmeans_run

python scripts/run_private_kmeans_inference.py \
  --run-dir /path/to/private/kmeans_run \
  --model /path/to/local/Qwen3-8B \
  --device cuda:0 --batch-size 4 --memory-fraction 0.30

python scripts/evaluate_private_kmeans.py \
  --run-dir /path/to/private/kmeans_run \
  --targets /path/to/private/targets.json
```

The memory fraction is the executed cap, not an automatically suitable value for every GPU. The K-means inference runner records model-file hashes and runtime settings, and rejects a mismatched resume configuration. Existing invalid outputs remain in the record.

If fresh embeddings are needed, `prepare_private_kmeans.py` accepts `--encoder /path/to/local/bge-base-zh-v1.5 --device cpu` instead of `--features-npz`. Record the actual representation and runtime. Optional `--reference-jobs` checks the regenerated real-arm prompts against a separately held private archive.

## Keeping source material private

The model scripts necessarily construct full prompt instances in the external run directory. Keep that directory out of Git and out of release attachments. The public release is produced from an allowlist of derived fields; do not copy a whole private run directory back into this checkout.
