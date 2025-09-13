# Quickstart

This guide gets you from installation to first results.

## 1. Install

- Python 3.10–3.12
- Recommended: create a fresh environment

```bash
conda create -n nepkit python=3.10
conda activate nepkit
pip install NepTrainKit
```

Windows portable: download `NepTrainKit.win32.zip` from Releases and run the executable.

## 2. Launch

```bash
nepkit
# or
NepTrainKit
```

## 3. NEP Dataset Display

- Import data via the top‑left Open button or drag‑and‑drop.
- Supported imports:
  - `train.xyz` + corresponding `*.out` files
  - `nep.txt` (optional; uses NEP89 if absent) + `train.xyz`
  - DeepMD directory (auto‑detected)
- Interact with plots, search by `Config_type` or formula, select, delete, and export:
  - Export menu → “Export Selected Structures” for chosen frames
  - Save button exports `export_remove_model.xyz` and `export_good_model.xyz`

## 4. Make Dataset

- Drag structures (XYZ/POSCAR/CIF) into the window or use Open.
- Build a pipeline with cards; use groups to branch/merge; add FPS filter if needed.
- Export to `make_dataset.xyz` when done.
- Save/Load card configurations as JSON to reuse pipelines.

## 5. Data Management

- Organize datasets into Projects and Models (versions), with notes and tags.
- Right‑click for New/Modify/Delete, Open Folder, and Tag management.
- Press `Ctrl+F` for advanced search.

## 6. Settings

- Choose plotting force mode (Raw vs Norm) and canvas engine (PyQtGraph vs Vispy).
- NEP Backend: select CPU/GPU/Auto for NEP calculations; Auto tries GPU first and falls back to CPU
- GPU Batch Size: adjust the number of frames per GPU slice to balance speed and memory
- Enable Auto loading, adjust covalent radius threshold, sorting, and menu grouping.
- Check app updates and NEP89 model, open help and feedback.

Note:
- GPU backend requires a compatible NVIDIA driver and CUDA 12.4 runtime. If you see “CUDA driver version is insufficient for CUDA runtime version”, switch NEP Backend to CPU in Settings.

## 7. Tips

- Use Vispy for large scenes if your GPU supports OpenGL.
- Toggle formula search to match by composition rather than tags.
- Use the structure toolbar to export descriptors or mark non‑physical bonds.

