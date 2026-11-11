# Runbook

How to prepare the data, train and evaluate. Environment setup: **[installation.md](installation.md)**.
All commands: environment active (`(aneurysmgen_env)` in the prompt), run from the project folder.

```
python scripts/run_prepare_data.py     # 1. split + patches   (optional, training does it too)
python scripts/run_train.py            # 2. train
python scripts/run_eval.py             # 3. evaluate
```
Every script: `--help` lists all options.

---

## 0. Paths: nothing to set by hand

- **Project folder** = the parent of `scripts/`, found automatically (scripts also work when started from another folder).
  Notebooks search upwards from their folder for `src/aneurysm/`.
- Every other path is **relative to the project folder** and comes from a config file:

| What | Where it is set | Default |
|---|---|---|
| scans, masks, CSV | `configs/data/data.yaml`: `scans_subdir`, `masks_subdir`, `instances_csv` | `Dataset/...` |
| split + patch cache | `configs/data/data.yaml`: `work_subdir` | `work/` |
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

## 1. Prepare the data (optional)

```bash
python scripts/run_prepare_data.py
python scripts/run_prepare_data.py --set patch_depth=64          # another patch setting -> its own folder
python scripts/run_prepare_data.py --overwrite                   # re-cut the patches of this setting
```
Writes `work/splits/splits.csv` (fixed train / val / test per case) and `work/patches/<setting>/` (one `.npz` per aneurysm + `index.csv`).
Training and evaluation call this automatically and reuse existing patches.

---

## 2. Train

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
python scripts/run_train.py --data-set patch_depth=24                        # smaller patches if memory is tight
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

## 3. Evaluate

Set the run to evaluate in `configs/eval/lefusion_eval.yaml` (`run_name`, or `run_dir`), or on the command line:
```bash
python scripts/run_eval.py --set run_name=pilot_64x64x32_small
python scripts/run_eval.py --set run_dir=outputs/pilot_64x64x32_small checkpoint=latest
python scripts/run_eval.py --set run_name=pilot_64x64x32_small max_patches=4 case_2_real_dilated=false
python scripts/run_eval.py --config configs/eval/my_eval.yaml
```
The model and data settings are read **from the checkpoint**, so they always match the trained run.
**Data needed:** only `work/` (the cached patches of the trained setting, e.g. copied from the training machine). `Dataset/` is needed only for `full_scan` (skipped with a message if the scan is missing) or when the patches are not cached yet.
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

## 4. Notebooks (same code)

| Notebook | Script |
|---|---|
| `01_dataset.ipynb` | `run_prepare_data.py` + data checks and plots |
| `02_train_lefusion.ipynb` | `run_train.py` (overrides in `TRAIN_OVERRIDES`) |
| `03_eval_lefusion.ipynb` | `run_eval.py` (overrides in `EVAL_OVERRIDES`, figures shown inline) |

Clear notebook outputs before committing (the repo is public).

---

## 5. Moving a run between computers

Copy `outputs/<run_name>/` to the other computer (same place, or anywhere and use `run_dir`).
The data must be there too (`Dataset/`, ideally `work/splits/splits.csv`); patches are re-cut automatically if `work/` is missing.

---

## 6. Run problems

| Problem | Fix |
|---|---|
| `Unknown keys in ...` | Typo in the YAML or in `--set`; the message lists the allowed keys |
| `No trained run at ...` | Wrong `run_name` / `run_dir`; the message lists the existing runs |
| `... not found. Available: [...]` | Wrong `checkpoint` name; pick one from the list |
| `already has a checkpoint made with different settings` | Network or patch settings changed: use a new `run_name` |
| Training very slow, GPU memory full + shared memory used | Sysmem fallback (see Windows GPU tips); use the small network or `--data-set patch_depth=24` |
| `LeFusion code not found` | Clone LeFusion into the project folder (installation.md, step 2) |
| `No patches in split 'val'` | Too few cases for a val split; check `val_frac` in `configs/data/data.yaml` |
