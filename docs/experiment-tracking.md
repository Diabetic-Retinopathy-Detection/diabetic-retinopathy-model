# Experiment Tracking

Training runs are logged to **MLflow** (run comparison) and **TensorBoard** (live curves). Both record the same metrics — this is intentional redundancy, not conflict.

## Quick start

```bash
# Run a training job (MLflow + TensorBoard enabled by default)
uv run dr-train --phase pretrain --device mps --seed 42

# View MLflow runs
mlflow ui
# → http://localhost:5000

# View TensorBoard curves
tensorboard --logdir logs/
# → http://localhost:6006
```

## What gets logged

### MLflow

Logged once at run start:

- Scalar values from `Settings` as params. With the current default settings,
  this is 48 parameters; an optional configured string such as
  `finetune_checkpoint` can increase the count to 49. Paths, lists, tuples,
  `None` values, and the computed `model_name` are not logged as params.
- **Git commit hash** as tag
- **CLI command** as tag

Logged per epoch:

| Metric | Description |
|---|---|
| `loss/contrastive` | MoCo v3 InfoNCE loss |
| `loss/saliency` | Saliency segmentation BCE loss |
| `loss/total` | Weighted sum of both losses |
| `lr` | Current learning rate |
| `momentum_m` | Current EMA momentum value |

### TensorBoard

Pretraining logs the five loss/optimization metrics above plus `time/epoch`
and `time/total`. Fine-tuning logs `loss/train`, `loss/val`, `kappa/val`, `lr`,
and `time/epoch` when validation is enabled. Useful for watching training
progress in real time.

## Configuration

Add to your YAML config:

```yaml
mlflow: true                              # enable/disable MLflow
mlflow_tracking_uri: null                 # leave unset for MLflow's default URI
mlflow_experiment_name: dr-pretrain       # experiment name in MLflow UI
tensorboard: true                         # enable/disable TensorBoard
```

### Remote tracking server

To log to a remote MLflow server instead of the local SQLite database:

```yaml
mlflow_tracking_uri: http://mlflow-server:5000
```

### Disabling tracking

For faster iteration or debugging, disable both:

```yaml
mlflow: false
tensorboard: false
```

## Viewing runs

### MLflow UI

```bash
mlflow ui
```

Opens a web interface at `http://localhost:5000` with:

- **Compare runs** — select multiple runs and compare params/metrics side-by-side
- **Run details** — view all logged params, metrics, and tags for a single run
- **Metric plots** — time-series plots of all logged metrics
- **Filter runs** — search by param values (e.g., `batch_size = 32`)

### TensorBoard

```bash
tensorboard --logdir logs/
```

Opens a web interface at `http://localhost:6006` with:

- **SCALARS** tab — loss curves, learning rate, momentum
- Real-time updates during training

### dr-report (PDF export)

Export TensorBoard scalars to a PDF:

```bash
uv run dr-report --logdir logs/pretrain_vit_p16_e768_d12_h12_c5/
uv run dr-report --logdir logs/pretrain_vit_p16_e768_d12_h12_c5/ --output report.pdf
```

## Local files

When an explicit SQLite tracking URI is configured, MLflow stores data in a
database and artifact location such as:

| File | Purpose |
|---|---|
| `mlflow.db` | SQLite database with run metadata |
| `mlruns/` | Run artifacts and metric history |

The training code does not automatically select this SQLite layout when
`mlflow_tracking_uri` is `null`; configure a URI explicitly when a shared or
portable local database is required. To reset a configured local store, delete
its database and artifact directory:

```bash
rm -rf mlflow.db mlflow.db-shm mlflow.db-wal mlruns/
```
