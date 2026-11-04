# Installation

Everything uses **Python 3.10 + `venv` + `pip`** (no conda).
The virtual environment lives in the project folder as **`venv/`**.

- **Windows + NVIDIA GPU**: training (`02_train_lefusion.ipynb`)
- **macOS**: data preparation (`01_dataset.ipynb`), no GPU training

> **About moving `venv/` between computers:** a venv stores the full path of the
> Python it was made with and contains OS-specific files. A venv made on the Mac
> will **not** run on Windows, and usually not on another Windows PC either.
> When you move the project, keep `venv/` if you like, but on the new machine
> **delete it and re-create it** with the steps below (5 minutes, same
> packages, because `requirements.txt` pins every version).

---

## 1. Before you start

| Need | Windows (training) | macOS (data prep) |
|---|---|---|
| Python **3.10** | <https://www.python.org/downloads/release/python-31011/> → "Windows installer (64-bit)". Tick **"Add python.exe to PATH"**. | Same page → "macOS 64-bit universal2 installer" |
| NVIDIA driver | Recent driver (version **530 or newer**, for CUDA 12.1). Check with `nvidia-smi` | not needed |
| Disk space | ~6 GB for packages + data | ~3 GB + data |

You do **not** need to install CUDA separately: the PyTorch wheel includes it.

Check Python:

**Windows (PowerShell)**
```powershell
py -3.10 --version
```
**macOS (Terminal)**
```bash
python3.10 --version
```
Both should print `Python 3.10.x`.

---

## 2. Get the code and data

Folder layout expected by the code:
```
Aneurysm_dataset/
├── 01_dataset.ipynb
├── 02_train_lefusion.ipynb
├── configs/               data.yaml, lefusion_train.yaml
├── src/aneurysm/          project code
├── requirements.txt
├── installation.md
├── LeFusion/              LeFusion code (git clone, see below)
└── Dataset/
    ├── CTA_scans/internal_Tr01/        Tr0001.nii.gz ...   (add internal_Tr02, ... later)
    ├── Annotations/internal_annotation/  Tr0001.nii.gz ... Ts0152.nii.gz
    ├── Annotations/external_annotation/
    └── internal_instances.csv
```

If `LeFusion/` is missing, get it (needs git):
```bash
git clone https://github.com/HINTLab/LeFusion.git
```
We only **import** its code; nothing in `LeFusion/` is changed or installed.

---

## 3. Create the virtual environment

Open a terminal **in the project folder** (`Aneurysm_dataset/`).

### Windows (PowerShell)
```powershell
cd "C:\path\to\Aneurysm_dataset"

# 1) create venv/ with Python 3.10
py -3.10 -m venv venv

# 2) activate it (prompt changes to "(venv) ...")
.\venv\Scripts\Activate.ps1
```
If PowerShell says *"running scripts is disabled"*, run this once, then activate again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
(Command Prompt instead of PowerShell: `venv\Scripts\activate.bat`)

### macOS (Terminal)
```bash
cd "/path/to/Aneurysm_dataset"

# 1) create venv/ with Python 3.10
python3.10 -m venv venv

# 2) activate it (prompt changes to "(venv) ...")
source venv/bin/activate
```

---

## 4. Install the packages

Same commands on both systems, **with the venv activated**:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
`requirements.txt` points pip to the PyTorch CUDA 12.1 builds. On macOS pip
automatically uses the normal macOS build instead.

### Check the install
```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import nibabel, torchio, einops, rotary_embedding_torch, omegaconf; print('imports OK')"
```
Expected on **Windows**: `torch 2.1.2+cu121 | CUDA: True`
Expected on **macOS**: `torch 2.1.2 | CUDA: False` (that is fine there)

**Windows shows `CUDA: False` or a version without `+cu121`?** pip installed the CPU build. Fix:
```bash
pip uninstall -y torch torchvision
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```
Then run the check again. If it is still `False`, update the NVIDIA driver (`nvidia-smi` must work).

---

## 5. Register the venv for Jupyter

So notebooks run with this venv's packages:
```bash
python -m ipykernel install --user --name aneurysm --display-name "Python (aneurysm venv)"
```

---

## 6. Run the notebooks

With the venv activated, in the project folder:
```bash
jupyter lab
```
In Jupyter, open a notebook and choose the kernel **"Python (aneurysm venv)"** (top right).

| Order | Notebook | Where | What it does |
|---|---|---|---|
| 1 | `01_dataset.ipynb` | Mac or Windows | finds scans, makes the split, cuts patches (`work/`), checks the DataLoaders |
| 2 | `02_train_lefusion.ipynb` | **Windows (GPU)** | trains LeFusion; outputs go to `outputs/<run_name>/` |

Notebook 2 prepares the data itself, so on Windows you can run it directly.
Settings are in `configs/data.yaml` and `configs/lefusion_train.yaml`.

**VS Code instead of Jupyter Lab:** open the folder, open a notebook, *Select Kernel* → *Python Environments* → `venv`.

---

## 7. Every time you come back

**Windows**
```powershell
cd "C:\path\to\Aneurysm_dataset"
.\venv\Scripts\Activate.ps1
jupyter lab
```
**macOS**
```bash
cd "/path/to/Aneurysm_dataset"
source venv/bin/activate
jupyter lab
```
Leave the venv with `deactivate`.

---

## 8. Moving the project to another computer

1. Copy the project folder (with `Dataset/`; `work/` and `outputs/` are optional — they are re-created).
2. On the new computer, delete the old venv and make a new one:
   - Windows: `Remove-Item -Recurse -Force venv`
   - macOS: `rm -rf venv`
3. Repeat steps 3 → 5.

Paths in `work/` (splits, patch index) are stored **relative to the project folder**, so they work on any computer. Keep `work/splits/splits.csv` when moving: it fixes which patient is in train / val / test.

---

## 9. Common problems

| Problem | Fix |
|---|---|
| `py` not found (Windows) | Re-run the Python installer and tick "Add python.exe to PATH", or use the full path, e.g. `C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe -m venv venv` |
| `No module named aneurysm` in a notebook | Start Jupyter **from the project folder**, and pick the "Python (aneurysm venv)" kernel |
| `LeFusion code not found` | `LeFusion/LeFusion/ddpm/diffusion.py` must exist: clone LeFusion into the project folder (step 2) |
| `Out of GPU memory` in the memory check | In `02_train_lefusion.ipynb` set `TRAIN_OVERRIDES = {"base_channels": 32}` |
| Very slow first run of notebook 1 | Normal: patches are cut once and cached in `work/patches/` |
| Install error about "invalid metadata" | Make sure `requirements.txt` has `hydra-core==1.3.2` and `omegaconf==2.3.0` (older versions fail with new pip) |
| Windows path errors with very long folder names | Keep the project in a short path, e.g. `C:\projects\Aneurysm_dataset` |
