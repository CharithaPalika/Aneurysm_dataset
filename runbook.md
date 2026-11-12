# Runbook

How to prepare the data, train and evaluate. Environment setup: **[installation.md](installation.md)**.
All commands: environment active (`(aneurysmgen_env)` in the prompt), run from the project folder.

```
python scripts/run_preprocess.py       # 1. preprocess the dataset (root in configs/data/sources/) -> work/
python scripts/run_smoke_test.py       # 2. smoke test: a few steps of train + eval (before every long run)
python scripts/run_train.py            # 3. train     (reads work/ only; one GPU)
python scripts/run_eval.py             # 5. evaluate  (reads work/ only)
```
Every script: `--help` lists all options.

### Two ways to run

**A. Preprocess first (recommended)**: preprocess once, then train and evaluate from the work dir.
```powershell
python scripts/run_preprocess.py          # dataset root -> work/  (needs the dataset)
python scripts/run_train.py               # preprocess: false (default) -> reads work/ only
python scripts/run_eval.py                # preprocess: false (default) -> reads work/ only
```

**B. Preprocess inside training**: skip the separate step; training cuts the missing patches first.
```powershell
python scripts/run_train.py --set preprocess=true      # or set  preprocess: true  in configs/train/lefusion_train.yaml
python scripts/run_eval.py                             # patches are now in work/, so preprocess: false is fine
```

| Setting (train / eval config) | Effect |
|---|---|
| `preprocess: false` (default) | use only the preprocessed data in the work dir; the raw dataset is not needed; stops with a clear message if the patches are missing |
| `preprocess: true` | cut missing patches from the raw dataset into the work dir first (existing ones are reused), then run |
| `work_dir: null` (default) | work dir = `work/` (`work_subdir` in `configs/data/data.yaml`); eval: the one used in training |
| `work_dir: <folder>` | use the preprocessed data in that folder, e.g. `D:/aneurysm_work` |

---

## 0. Paths: the only one to set is the dataset root

### The dataset (read-only, anywhere on disk)

Set the path once in **`configs/data/sources/zenodo_la.yaml`**, up to and including `Data/`:
```yaml
root: "C:/Users/<you>/Datasets/Zenodo/Data"       # Windows (forward slashes work)
# root: "/Volumes/Data/Zenodo/Data"               # macOS
```
Expected inside `Data/` (other files and the `external_*` folders are ignored):
```
Data/
├── internal_Tr01/ ... internal_Tr12/    Tr0001.nii.gz ...   -> train / val  (split by val_frac)
├── internal_Ts01/, internal_Ts02/       Ts0001.nii.gz ...   -> test
├── internal_annotation/                 masks, same file names as the scans
└── internal_instances.csv
```
- Only `.nii.gz` files are read; the `.nii` copies are ignored.
- The dataset folder is **never written**. Everything made from it goes to `work/` inside the project.
- Without editing the file: `--set sources.zenodo_la.root=PATH` (preprocess) or `--data-set sources.zenodo_la.root=PATH` (train).
- On another computer only `root` changes: scan paths are stored relative to `root`, patch paths relative to the project.

**Which datasets are used** is set in `configs/data/data.yaml`:
```yaml
datasets:
  zenodo_la: true
```
**Adding a dataset later** (e.g. from Kaggle): write a reader `src/aneurysm/data/sources/<name>.py`
(a `DatasetSource` subclass that returns the common case table) and register it in
`src/aneurysm/data/sources/__init__.py`, add `configs/data/sources/<name>.yaml` with its `root` and layout,
and set `<name>: true` under `datasets`. Each dataset is preprocessed into its own `work/<name>/`.

### Everything else: automatic

- **Project folder** = the parent of `scripts/`, found automatically (scripts also work when started from another folder).
  Notebooks search upwards from their folder for `src/aneurysm/`.
- Every other path is **relative to the project folder** and comes from a config file:

| What | Where it is set | Default |
|---|---|---|
| raw dataset (read-only) | `configs/data/sources/<name>.yaml`: `root` | must be set |
| preprocessed data (split + patches) | `configs/data/data.yaml`: `work_subdir`; train / eval can point elsewhere with `work_dir` | `work/<dataset>/` |
| training runs | `configs/train/lefusion_train.yaml`: `output_dir`, `run_name` | `outputs/<run_name>/` |
| run to evaluate | `configs/eval/lefusion_eval.yaml`: `run_name` **or** `run_dir` | `outputs/<run_name>/` |
| LeFusion code | fixed | `LeFusion/LeFusion/ddpm/` |

Absolute paths also work in the YAML files (e.g. data on another drive).

**Config files and overrides** (same for all scripts):
- `--config PATH` (and `--data-config PATH` for training): use another YAML file. Relative paths work from the current folder or the project folder. Without it, the defaults in `configs/` are used.
- `--set key=value ...` (and `--data-set ...`): change single values for this run only. Values are read like YAML (`32`, `true`, `null`, `[1,2,4,4]`). Typos stop the script with the list of allowed keys.
- **PowerShell:** put values with `[ ]` in quotes: `"dim_mults=[1,2,4,4]"`.
- For settings you keep, make a copy of the YAML (e.g. `configs/train/pilot_small.yaml`) and pass it with `--config`.

---

## 1. Preprocess

```bash
python scripts/run_preprocess.py                                  # uses configs/data/data.yaml
python scripts/run_preprocess.py --data-config configs/data/my_data.yaml
python scripts/run_preprocess.py --set patch_depth=64             # another patch setting -> its own folder
python scripts/run_preprocess.py --set work_subdir=D:/aneurysm_work   # save somewhere else
python scripts/run_preprocess.py --set sources.zenodo_la.root=D:/Datasets/Zenodo/Data   # dataset root, no file edit
python scripts/run_preprocess.py --overwrite                      # re-cut the patches of this setting
```
Needs the dataset root (section 0). Writes, per dataset:
```
work/zenodo_la/splits.csv                          case -> train / val / test (fixed; new cases are added)
work/zenodo_la/patches/<setting>/<split>/*.npz     one patch per aneurysm, resampled to spacing_mm
work/zenodo_la/patches/<setting>/index.csv         one row per patch
work/zenodo_la/patches/<setting>/processed_cases.txt   cases done (also normal scans without patches)
```
Split: `internal_Tr*` cases -> train / val (`val_frac` in `configs/data/data.yaml`, by patient, stable);
`internal_Ts*` cases -> test. **Validation during training uses the val split**; test is only for the final numbers.
Normal scans (no aneurysm) give no patches. The first run over all ~1,300 scans takes a long time; it can be
stopped and restarted (finished cases are skipped).
`<setting>` encodes patch size, spacing, HU window, margin and min size, e.g. `64x64x32_sp0.45_hu-100to900_m8_min20`.

**Data source in training and evaluation** (in their configs, or `--set`):

| `preprocess` | `work_dir` | What happens |
|---|---|---|
| `false` (default) | `null` | use the preprocessed data in `work/`; the raw dataset is not needed |
| `false` | a folder | use the preprocessed data in that folder |
| `true` | any | cut missing patches from the raw dataset first (needs `root`), then run |

**If you change preprocessing settings** (`configs/data/data.yaml`): run the preprocess again with the new file;
it creates a new `<setting>` folder (old ones are kept). Training must use the **same data config**
(`--data-config`, default the same file), otherwise it stops with "No complete preprocessed data for
patch setting ...". Evaluation needs nothing: it takes the setting of the trained model from the checkpoint.

---

## 2. Smoke test (before every long run)

A few minutes of the **real** pipeline with the **same configs** as the real run (network size, batch,
GPUs, data), but only a few steps and its own folder `outputs/smoke_tests/smoke_<date>_<time>/`
(your real run folder is never touched).

```bash
python scripts/run_smoke_test.py                                                  # pilot config, 1 GPU
python scripts/run_smoke_test.py --config configs/train/lefusion_train_a100.yaml  # cluster config, 1 GPU
torchrun --standalone --nproc_per_node=4 scripts/run_smoke_test.py --config configs/train/lefusion_train_a100.yaml
python scripts/run_smoke_test.py --tiny                                           # tiny network: laptop / CPU, pipeline only
```
Options: `--steps N` (default 20), `--set` / `--data-set` as for training, `--skip-eval`, `--name`.

It checks, and prints `[PASS]` / `[FAIL]` for each:

| Check | What it proves |
|---|---|
| training phase 1 reached its steps | data loads, model builds, fits in memory (memory-check line), loss goes backward |
| resume from latest.pt (phase 2) | Ctrl+C / job restarts will continue correctly |
| train loss logged and finite | no NaN / inf with this batch and AMP |
| validation loss computed | val split exists and works |
| checkpoint latest.pt / best.pt written | outputs folder is writable |
| preview sample PNG written | generation (RePaint) works with this network |
| evaluation (case 1 + case 4, one patch) | the eval code works on a checkpoint of this config |

Ends with `SMOKE TEST PASSED` (exit code 0) or `SMOKE TEST FAILED` (exit code 1).
Before the real run, read from its output:
- **Memory check** line: GPU memory needed per GPU. Leave a margin; if it is close to 40 GB, lower `batch_size`
  and raise `grad_accum` (keep `batch_size x grad_accum x GPUs` the same).
- **s/step**: multiply by `max_steps` for the run time (the first steps are slower).
- `--tiny` only checks that the code runs (8 channels, 20 diffusion steps); it says nothing about memory or speed.

Smoke folders can be deleted afterwards (`outputs/smoke_tests/`).

---

## 3. Train (one GPU)

```bash
python scripts/run_train.py                                       # configs/train/lefusion_train.yaml + configs/data/data.yaml
```
Small network for the 4 GB Quadro T2000 (same as `02_train_lefusion.ipynb`):
```powershell
python scripts/run_train.py --set run_name=pilot_64x64x32_small base_channels=32 "dim_mults=[1,2,4,4]" attn_heads=4
```
Other examples:
```bash
python scripts/run_train.py --config configs/train/pilot_small.yaml          # your own copy of the config
python scripts/run_train.py --data-set patch_depth=24 --set preprocess=true  # smaller patches, cut them first
python scripts/run_train.py --set work_dir=D:/aneurysm_work                  # preprocessed data in another folder
python scripts/run_train.py --set run_name=lr2e4 lr=2e-4 max_steps=20000
```

**What happens**
- Output folder `outputs/<run_name>/`:
  ```
  config.yaml            settings used (config_resume_*.yaml when resumed)
  train_log.txt          everything printed
  metrics.csv            train / val losses per step
  samples/step_*.png     preview generations
  checkpoints/           latest.pt, best.pt (lowest val loss), step_*.pt (last keep_last)
  ```
- Printed every `log_every` steps (loss, speed), every `eval_every` (val loss), every `sample_every` (preview PNG).
- At the start a **memory check** prints the peak GPU memory; it stops with a clear error if the model does not fit.

**Stop / resume / new run**
- **Ctrl+C** saves `checkpoints/latest.pt`. Running the **same command** again resumes (`resume: true`).
- A new `run_name` starts a fresh run. An existing run is never overwritten.
- Changing the network size or patch setting and reusing the same `run_name` stops with "already has a checkpoint made with different settings": use a new `run_name`.
- Same settings + same `seed` give the same run.

**Windows GPU tips**
- NVIDIA Control Panel → *Manage 3D settings* → **CUDA - Sysmem Fallback Policy: Prefer No Sysmem Fallback**. Otherwise a too-large model silently spills into system RAM and becomes extremely slow instead of failing.
- In the terminal `num_workers: 2` (the default on Windows) works; in Jupyter use `0`.
- Watch the GPU: `nvidia-smi -l 5` in a second terminal.


---

## 4. Train on the cluster (4 x A100 40 GB, multi-GPU)

Config: **`configs/train/lefusion_train_a100.yaml`**: the network of the LeFusion paper / repo, which
trains on the same patch size as ours (32 x 64 x 64):

| Setting | Value | Note |
|---|---|---|
| `base_channels`, `dim_mults` | 64, [1, 2, 4, 8] | channels 64 / 128 / 256 / 512, about 36 M parameters |
| `attn_heads`, `attn_dim_head` | 8, 32 | |
| `timesteps`, `loss_type`, `lr` | 300, l1, 1e-4 | |
| `batch_size` | 5 **per GPU** | 4 GPUs -> 20 per step (paper: 20) |
| `grad_accum` | 2 | effective batch 5 x 2 x 4 = **40** (paper: 40) |
| `max_steps` | 50,000 | paper: 50k; README: good results after 20k; ~3 days on 4 x A100 (check s/step in the smoke test) |
| `amp` | true | the paper used fp32; AMP is about 2x faster on A100 |
| `num_workers` | 6 per GPU | about the CPU cores per GPU minus 2 |

**Steps on the cluster**
```bash
# 0. once: environment (installation.md) + dataset root
#    configs/data/sources/zenodo_la.yaml -> root: "/path/on/cluster/Data"
# 1. preprocess (CPU job, hours; restartable)
python scripts/run_preprocess.py --threads 4
# 2. smoke test on the GPUs (minutes)
torchrun --standalone --nproc_per_node=4 scripts/run_smoke_test.py --config configs/train/lefusion_train_a100.yaml
# 3. real run (same command again after a stop or time limit resumes from latest.pt)
torchrun --standalone --nproc_per_node=4 scripts/run_train.py --config configs/train/lefusion_train_a100.yaml
# 4. evaluate (one GPU is enough)
python scripts/run_eval.py --set run_name=a100_paper_64x64x32
```

**How multi-GPU works here** (PyTorch DistributedDataParallel; `torchrun` starts one process per GPU)
- `batch_size` is **per GPU**; effective batch = `batch_size x grad_accum x GPUs` (printed at the start).
- Each GPU trains on a different part of the training patches each epoch; gradients are averaged.
- Only the first process (rank 0) prints, writes logs / checkpoints / samples and computes the validation loss;
  the others wait meanwhile.
- `preprocess: true` with several GPUs: only rank 0 cuts patches, the others wait.
- Without `torchrun` the same script uses one GPU (same as before).
- The run is seeded, but multi-GPU runs are not bit-for-bit identical between runs (GPU arithmetic order).
- Number of GPUs can change between restarts, but keep `batch_size x grad_accum x GPUs` the same.

**Example SLURM job** (adapt partition / account / module names to your cluster):
```bash
#!/bin/bash
#SBATCH --job-name=lefusion_a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

cd /path/to/Aneurysm_dataset
source aneurysmgen_env/bin/activate
torchrun --standalone --nproc_per_node=4 scripts/run_train.py --config configs/train/lefusion_train_a100.yaml
```
Submit with `sbatch train_a100.sbatch`. If the time limit stops the job, submit it again: it resumes from `latest.pt`
(at most `save_every` = 1000 steps are lost). Run the smoke test the same way first (replace `run_train.py`
with `run_smoke_test.py`, `--time=00:30:00`).

**Cluster checks**
- `nvidia-smi` on a GPU node: driver must support CUDA 12.1 (`requirements.txt` installs torch 2.1.2 + cu121).
- Put the project (or at least `work/` via `work_dir`) on fast storage; the dataset root can stay read-only.
- `python -c "import torch; print(torch.cuda.device_count())"` on the GPU node should print 4.

---

## 5. Evaluate

Set the run to evaluate in `configs/eval/lefusion_eval.yaml` (`run_name`, or `run_dir`), or on the command line:
```bash
python scripts/run_eval.py --set run_name=pilot_64x64x32_small
python scripts/run_eval.py --set run_dir=outputs/pilot_64x64x32_small checkpoint=latest
python scripts/run_eval.py --set run_name=pilot_64x64x32_small max_patches=4 case_2_real_dilated=false
python scripts/run_eval.py --config configs/eval/my_eval.yaml
python scripts/run_eval.py --set run_name=pilot_64x64x32_small work_dir=D:/aneurysm_work
```
The model and data settings are read **from the checkpoint**, so they always match the trained run.
**Data needed:** only the preprocessed work dir with the patches of the trained setting (e.g. copied from the training machine). The raw dataset (`root` in `configs/data/sources/zenodo_la.yaml` on *this* machine) is needed only for `full_scan` (skipped with a message if it is not set) or with `preprocess: true`.
If the run is not found, the error lists the runs in `outputs/`.

**Checks** (each `true` / `false` in the config)

| Switch | What |
|---|---|
| `training_curves` | train / val loss from `metrics.csv` |
| `case_1_real_actual` | noisy real + actual mask |
| `case_2_real_dilated` | noisy real + dilated mask |
| `case_3_pseudo_actual` | noisy pseudo-healthy + actual mask |
| `case_4_pseudo_dilated` | noisy pseudo-healthy + dilated mask |
| `histogram_control` | same patch with low / medium / high contrast histograms |
| `copy_check` | case-1 results vs the closest training patch (needs case 1) |
| `full_scan` | paste one case-1 result into its CT scan, saved as NIfTI (needs case 1) |

Other settings: `checkpoint` (`best` / `latest` / `step_005000`), `split` (`val` while tuning, `test` only once at the end), `max_patches`, `seed`, `dilate_voxels`.

**Output** `outputs/<run_name>/eval/<checkpoint>_<split>/`:
`eval_config.yaml` (copy of the settings), `summary.csv` (one row per case), `case_*.png/.csv`,
`pseudo_healthy.csv`, `training_curves.png`, `histogram_control.png`, `copy_check.csv`, `full_scan/`.
Figures: yellow = actual mask, red = dilated mask, differences in HU.

**Time** on the T2000: about 0.5-1.5 min per generated patch; the default (8 patches x 4 cases + histogram control) takes roughly 20-60 min. Use `max_patches` or switch checks off to go faster.

---

## 6. Notebooks (same code)

| Notebook | Script |
|---|---|
| `01_dataset.ipynb` | `run_preprocess.py` + data checks and plots |
| `02_train_lefusion.ipynb` | `run_train.py` (overrides in `TRAIN_OVERRIDES`) |
| `03_eval_lefusion.ipynb` | `run_eval.py` (overrides in `EVAL_OVERRIDES`, figures shown inline) |

Clear notebook outputs before committing (the repo is public).

---

## 7. Moving a run between computers

Copy `outputs/<run_name>/` to the other computer (same place, or anywhere and use `run_dir`).
Copy the preprocessed work dir too (`work/`, or at least `work/<dataset>/splits.csv` and `work/<dataset>/patches/<setting>/`); the raw dataset is then only needed for `full_scan` (set its `root` on that machine).

---

## 8. Run problems

| Problem | Fix |
|---|---|
| Smoke test `[FAIL] training runs (... out of memory ...)` | Lower `batch_size` per GPU, raise `grad_accum` (same effective batch) |
| torchrun hangs at the start | All GPUs visible? (`CUDA_VISIBLE_DEVICES`, `torch.cuda.device_count()`); use `--standalone` on one node |
| `Unknown keys in ...` | Typo in the YAML or in `--set`; the message lists the allowed keys |
| `No trained run at ...` | Wrong `run_name` / `run_dir`; the message lists the existing runs |
| `... not found. Available: [...]` | Wrong `checkpoint` name; pick one from the list |
| `already has a checkpoint made with different settings` | Network or patch settings changed: use a new `run_name` |
| Training very slow, GPU memory full + shared memory used | Sysmem fallback (see Windows GPU tips); use the small network or `--data-set patch_depth=24` |
| `LeFusion code not found` | Clone LeFusion into the project folder (installation.md, step 2) |
| `No complete preprocessed data for patch setting ...` | Run `scripts/run_preprocess.py` with the same data config, fix `work_dir`, or set `preprocess: true` |
| `Dataset 'zenodo_la': set root ...` / `root folder not found` | Set `root` in `configs/data/sources/zenodo_la.yaml` to the `Data/` folder |
| `No patches in split 'val'` | Too few cases for a val split; check `val_frac` in `configs/data/data.yaml` |
