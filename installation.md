# Installation (environment)

**Python 3.10 + `venv` + `pip`** (no conda). The environment lives in the project folder as `aneurysmgen_env/`.
How to run data preparation, training and evaluation: see **[runbook.md](runbook.md)**.

| Machine | Used for |
|---|---|
| Windows + NVIDIA GPU | training and evaluation |
| macOS | data preparation, looking at results (no GPU training) |

---

## 1. Check the prerequisites

```powershell
# Windows (PowerShell)
py -3.10 --version          # -> Python 3.10.x
nvidia-smi                  # driver 530+ (CUDA 12.1 or newer shown top right)
```
```bash
# macOS (Terminal)
python3.10 --version        # -> Python 3.10.x
```
- Python 3.10: <https://www.python.org/downloads/release/python-31011/> (Windows: tick **"Add python.exe to PATH"**).
- CUDA does **not** need a separate install; the PyTorch wheel includes it.
- **Conda users:** if the prompt shows `(base)`, run `conda deactivate` first (or once:
  `conda config --set auto_activate_base false`), so `python` is the venv's and not conda's.

---

## 2. Get the code, LeFusion and the data

```bash
git clone <your repo URL> Aneurysm_dataset
cd Aneurysm_dataset
git clone https://github.com/HINTLab/LeFusion.git     # only if LeFusion/ is missing
```
The dataset can be anywhere (it is only read, never written). Set its path once in
`configs/data/sources/zenodo_la.yaml`:
```yaml
root: "C:/Users/<you>/Datasets/Zenodo/Data"     # the Data/ folder with internal_Tr*, internal_Ts*, internal_annotation/
```
Details and other datasets: [runbook.md](runbook.md), section 0.

---

## 3. Create and activate the environment

```powershell
# Windows (PowerShell), in the project folder
py -3.10 -m venv aneurysmgen_env
.\aneurysmgen_env\Scripts\Activate.ps1
# "running scripts is disabled"?  run once, then activate again:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
```bash
# macOS (Terminal), in the project folder
python3.10 -m venv aneurysmgen_env
source aneurysmgen_env/bin/activate
```
The prompt now starts with `(aneurysmgen_env)`.

---

## 4. Install the packages (requirements.txt) and the project code

Same on both systems, **with the environment active**:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt       # all libraries, pinned versions (PyTorch CUDA 12.1 build on Windows)
pip install -e .                      # our code in src/aneurysm (pyproject.toml); -e = code edits apply directly
```

---

## 5. Check the install

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import nibabel, torchio, einops, rotary_embedding_torch, omegaconf; print('libraries OK')"
python -c "import aneurysm.training, aneurysm.evaluation; print('project code OK')"
python -c "from pathlib import Path; print('LeFusion OK' if Path('LeFusion/LeFusion/ddpm/diffusion.py').exists() else 'LeFusion MISSING')"
python scripts/run_train.py --help
```
Expected: Windows `torch 2.1.2+cu121 | CUDA: True`; macOS `torch 2.1.2 | CUDA: False` (fine there).

Windows shows `CUDA: False` or no `+cu121` (CPU build was installed):
```bash
pip uninstall -y torch torchvision
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## 6. Jupyter kernel (only for the notebooks)

```bash
python -m ipykernel install --user --name aneurysmgen_env --display-name "Python (aneurysmgen_env)"
```
Then pick the kernel **"Python (aneurysmgen_env)"** in Jupyter / VS Code.

---

## 7. Every time you come back

```powershell
cd "C:\path\to\Aneurysm_dataset"; .\aneurysmgen_env\Scripts\Activate.ps1      # Windows
```
```bash
cd "/path/to/Aneurysm_dataset" && source aneurysmgen_env/bin/activate          # macOS
```
Leave with `deactivate`.

---

## 8. Update or rebuild the environment

```bash
pip install -r requirements.txt        # after requirements.txt changed
pip install -e .                       # after pyproject.toml changed (not needed for code edits)
```
Rebuild from scratch (also needed on a **new computer**: a venv cannot be copied between machines):
```powershell
deactivate; Remove-Item -Recurse -Force aneurysmgen_env     # Windows, then repeat steps 3-5
```
```bash
deactivate; rm -rf aneurysmgen_env                          # macOS, then repeat steps 3-5
```
When moving the project, keep `work/splits/splits.csv` (it fixes which patient is in train / val / test).

---

## 9. Install problems

| Problem | Fix |
|---|---|
| `py` not found (Windows) | Re-install Python with "Add python.exe to PATH", or use the full path `C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe -m venv aneurysmgen_env` |
| `No module named aneurysm` | Environment active? Run `pip install -e .` in the project folder |
| Wrong Python is used (conda) | `conda deactivate`, then activate `aneurysmgen_env` again; check with `python -c "import sys; print(sys.executable)"` |
| "invalid metadata" during install | `requirements.txt` must have `hydra-core==1.3.2` and `omegaconf==2.3.0` |
| Long-path errors on Windows | Keep the project in a short path, e.g. `C:\projects\Aneurysm_dataset` |
