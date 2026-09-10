# Staged localization learning

For observed labels, independent postprocessing CV and frozen external evaluation, see the [scientific validation contract](../docs/localize-scientific-validation.md). Historical CV postprocessing tuning metrics are not independent pipeline accuracy.

`cdskit localize-learn` supports a frozen ESM2 encoder with a trained multilabel
teacher head, saved teacher probabilities, and a compact CNN student. The exported
student runs on CPU through `cdskit localize`. This workflow does not establish an
accuracy improvement; compare the distilled student and matched non-distilled
control on held-out data before selecting a model.

Install the ML extra as described in [Testing](../TESTING.md). JSON and YAML
configurations share the same schema; see [the YAML example](../examples/localize-pipeline/config.yaml).
Replace its revision placeholder with an immutable encoder commit SHA, or set
`model_name` to a local ESM checkpoint directory containing safetensors and tokenizer
files. No remote model code is executed. `local_files_only: true` disables encoder
downloads. Teacher training freezes the encoder; it does not fine-tune ESM weights.
Encoder loading preserves the training RNG state, so an empty or populated
embedding cache does not change teacher initialization or dropout randomness.

## Partitions and configuration

Supply a UTF-8 TSV with these columns:

```tsv
accession	sequence	localization_labels	split	cluster_id
train1	MALWMRLLPLL	extracellular	train	cluster1
val1	MSSKSKPKDPS	nucleus;cytoplasm	validation	cluster2
test1	MLSLRQSIRFF	mitochondrion	test	cluster3
```

This illustrates the format, not a useful training dataset. Sequences are proteins;
labels are semicolon-separated. The default ordered labels are `nucleus`,
`cytoplasm`, `extracellular`, `mitochondrion`, `cell_membrane`, `endoplasmic_reticulum`,
`chloroplast`, `golgi_apparatus`, `lysosome_vacuole`, and `peroxisome`. An explicit `labels` list
can select a subset. An empty label field means all labels are negative.
`ensure_one_label` defaults to `false`, so thresholded predictions may also be
empty. Set it to `true` only when at least one of the selected labels must apply.
Teacher and student exports use the same setting. Column names can be mapped with `data.id_col`, `sequence_col`, `label_col`, `split_col`,
and `cluster_col`. Cluster IDs may be omitted entirely; if supplied, every row
must have one.

Assign train/validation/test partitions upstream using homology clusters. The
pipeline rejects duplicate IDs, overlapping canonical sequences or cluster IDs
between partitions, unknown labels, and incomplete cluster assignments. It does
not infer homology. Train and validation must be nonempty; test is optional until
evaluation. Only training rows receive teacher probabilities for distillation.
Early stopping and F1 threshold calibration use validation, and final metrics use
test. Labels without both positive and negative validation examples retain 0.5.

All configuration keys are checked; duplicate keys and invalid types fail.
Relative input, encoder and residue-cache paths resolve against the configuration
file. Encoder, residue-cache and run directories must not overlap (including
parent/child directories). Keep the configuration and data outside the run
directory. Configuration controls the number of PyTorch threads, teacher pooling/windowing, and student architecture/training;
legacy training options cannot be combined with staged settings.

## Execute and resume

```bash
cdskit localize-learn --stage teacher --config config.yaml --run_dir teacher-run
cdskit localize-learn --stage predict --config config.yaml --run_dir teacher-run
cdskit localize-learn --stage distill --config config.yaml --run_dir teacher-run
cdskit localize-learn --stage evaluate --config config.yaml --run_dir teacher-run
```

`--stage all` runs these stages in order, skipping evaluation if there is no test
partition. `--stage train` is the default and preserves the existing training CLI.
The teacher and predict stages need the ESM encoder. Distillation and evaluation
use PyTorch, saved probabilities and CNN weights without loading ESM.

For GPU teacher training followed by a separate CPU experiment, copy the complete
teacher run (including `run.json` and both completed stage directories) and the
partition TSV to the CPU machine. Create a CPU configuration with the same label
order and canonical train/validation rows, set `student.device: cpu`, and run:

```bash
cdskit localize-learn --stage all --config cpu.yaml --run_dir cpu-run --teacher_run teacher-run
cdskit localize --seq_file proteins.fa --seq_type protein --model cpu-run/distill/student.pt --report predictions.tsv
```

With `--teacher_run`, `all` runs distillation and evaluation only. The CPU
configuration may omit the `teacher` section entirely; it does not need the source
encoder files. `--teacher_run` is also accepted for `distill` and `evaluate`.
`train_control: true` trains a second CNN with the same settings and seed using
hard labels alone. Class balancing uses the negative/positive count ratio where
both outcomes exist, and neutral weight 1 where only one outcome exists. The student
mixes hard-label and teacher-probability binary cross entropy according to `distillation_weight` (0 = hard labels only).

The run records normalized configuration, data/partition/encoder/code hashes,
package versions and per-stage dependency/output hashes. Completed stages are
verified and reused by default; `--resume no` refuses existing stage outputs.
Changed inputs or configuration require a new run directory. Inputs are checked
again before each newly completed stage is published; changes during training
abort that stage. Each stage publishes its directory only after success. An interrupted stage restarts from its beginning;
epoch/optimizer continuation is not supported. A killed process may leave a lock
and hidden temporary stage directory: verify that its recorded host/PID has stopped
before manually removing these. The tool never takes over another process's lock.
Keep source runs unchanged while dependent runs execute. Manifests check accidental
changes, not the authenticity of artifacts from untrusted sources.

## Artifacts

| Path | Contents |
| --- | --- |
| `run.json` | Configuration and provenance |
| `teacher/model.pt` | Frozen-encoder configuration and trained teacher head |
| `predict/probabilities.npz` | Training probabilities, row IDs, sequence and teacher hashes |
| `distill/student.pt` | Self-contained CNN student for safe CPU loading |
| `distill/control.pt` | Matched hard-label CNN, when enabled |
| `evaluate/metrics.json` | Held-out classification and probability metrics |
| `evaluate/student.npz`, `control.npz` | Aligned test targets, predictions and probabilities |
| Each stage's `stage.json` | Dependency hashes, output hashes and execution environment |

Keep experiment-specific comparisons and benchmark outputs outside the repository.
SSH, Slurm resource requests, environment provisioning and job submission remain
outside CDSKIT; run the same commands inside an approved GPU allocation. No scheduler
partition, GPU resource syntax or remote machine policy is assumed by this workflow.

[The Slurm job wrapper](../scripts/localize_pipeline_job.sh) also works as a local
shell wrapper. Set `CDSKIT_PYTHON` to the absolute Python path in the environment
where this checkout is installed, then pass normal staged CLI arguments. For
example, inside an existing GPU allocation:

```bash
export CDSKIT_PYTHON=/absolute/path/to/environment/bin/python
bash scripts/localize_pipeline_job.sh --stage teacher --config config.yaml --run_dir teacher-run
```

For `sbatch`, supply your site's approved partition and resource options before
`scripts/localize_pipeline_job.sh`. Wait for teacher completion before submitting
`predict`, and for predict completion before `distill`, or use `--stage all` within
one suitable allocation. The wrapper itself requests no GPU or partition. Per-epoch
validation BCE and the selected epoch are saved in `teacher/training.json` and
`distill/student-training.json` (plus the control equivalent when enabled).
Teacher training also records validation macro average precision.

Teacher experiments can set `teacher.loss` to `bce` (default), `weighted_bce`, or
`asl`. Weighted BCE uses square-root negative/positive ratios from training labels
only, capped at 10; labels missing either class retain weight 1. ASL uses positive
focusing exponent 1, negative exponent 4, negative probability clipping 0.05, and
detached focusing weights. Validation BCE is always unweighted and comparable
across training losses.

`teacher.selection_metric` accepts `bce` (default, minimized) or `macro_ap`
(maximized over validation labels with positive examples). Thresholds are still
calibrated on validation data only. The selected loss and metric are recorded in
the model. Do not select recipes or epochs from test results.

`teacher.pooling` accepts `mean`, `light_attention`, `label_attention`, and
`terminal_attention`. The last adds mean ESM embeddings of the first and last 32
residues to the light-attention representation. Short sequences use their actual
residues; padding is excluded. The benchmark CLI exposes the same choices through
`--plm_pooling`, `--plm_loss`, and `--plm_selection_metric`. These are experiment
options; their presence does not establish an accuracy improvement.

## Feature and decision versions

See the [localization feature and decision contract](../docs/localize-scientific-contract.md)
for the corrected nine-residue PTS2 definition, legacy model compatibility,
`--decision_policy safe-v1`, v2 output and explicit taxonomy constraints.
Published artifacts retain their historical defaults; new staged pipeline models
default to safe inference with `ensure_one_label=False`. Scores remain unverified
as calibrated biological probabilities.
