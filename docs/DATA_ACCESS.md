# Data and access

## Public content

- Analysis and model-execution code.
- Prompt instructions and templates with empty source-content fields.
- Newly assigned post identifiers and question identifiers, preserving selection and grouping relationships.
- Numeric-only model response strings, normalized predictions and validity indicators.
- Aggregate semantic predictions, question-level JS scores and paired statistical summaries.
- Grouping diagnostics, random seeds, generation settings, software versions and public model weight hashes.
- SHA-256 hashes of the files in this public release.

The original response strings were checked to contain only numeric arrays or a single `distribution` array, optionally wrapped in a JSON code fence. All 1,300 passed this content check. The invalid all-zero response is retained and marked invalid.

## Source data kept outside the repository

- Social-media post bodies, snippets, source URLs and platform/author identifiers.
- The mapping between source post identifiers and public post identifiers.
- Survey question text, answer-option wording, respondent records, answer counts and target answer distributions.
- Complete prompt instances containing posts or survey text.
- Embedding arrays, per-post semantic probability caches and model weights.
- Data-bearing annotation forms, private source keys and uncompleted human-label files.
- Private filesystem paths, server/user details and credentials.

No raw dataset or access credential is bundled. New post IDs are randomly assigned opaque labels; they are not hashes of post text or original platform identifiers. Question IDs provide a consistent link between the public score tables and output files, without the original questionnaire text.

## Reproduction levels

1. **Public statistics:** use the released per-question scores to reproduce paired differences, exact sign-flip p values, descriptive bootstrap intervals and the specified Holm family. No private inputs are needed.
2. **Output parsing and grouping checks:** verify response parsing, failures, group sizes, post-set equality and numerical aggregation using public model outputs and anonymous assignments.
3. **Prediction-to-JS verification:** supply the authorized private target distributions using the public question IDs and documented option order.
4. **Model reruns:** supply the authorized text/question bundle and appropriate local model weights. Use the private execution guide; generated data remain outside the repository.

Access to underlying survey records and social-media material must be arranged separately with the relevant data holders. Publication of this code and derived output does not grant access to those source records. Collection permissions and ethics-review status must be documented from the underlying study records.

## Adding future files

The Git ignore rules are a convenience, not the release review. Before any later publication, regenerate the release manifest, run the tests and inspect newly staged files. Do not add a private run directory or attach a source-data archive to a GitHub release or issue.
