# Settings

Personalize behavior and performance trade‑offs. Changes take effect immediately and persist to your user config.

## Personalization

- Force data format: Choose how forces are plotted
  - Raw: Use 3N vector format grouped by atoms
  - Norm: Use per‑structure |F| (faster plotting and simpler visuals)
- Canvas Engine: Choose rendering backend
  - PyQtGraph (CPU): Default, broad compatibility
  - Vispy (GPU): Uses OpenGL, faster for large scenes (requires working GPU/driver)
- Auto loading: On startup, if `./train.xyz` and `./nep.txt` exist, auto‑load the dataset
- Covalent radius coefficient: Threshold for “non‑physical bond” detection in structure view
- Sort atoms: Normalize atom ordering when processing structures in Make Dataset
- Use card group menu: Group console cards by category in the add‑card menu

## NEP Settings

- NEP Backend: Select CPU, GPU, or Auto
  - Auto: Tries GPU first; falls back to CPU if GPU is unavailable
  - GPU: Requires a supported NVIDIA driver and CUDA 12.4 runtime; otherwise an in‑app warning will appear and the app switches to CPU
  - CPU: Always use the CPU backend
- GPU Batch Size: Number of frames per GPU slice when running NEP calculations
  - Larger batches improve throughput but use more GPU memory
  - Reduce if you encounter out‑of‑memory errors

## About

- About NEP89: Check and download the official NEP89 model
- Help: Open documentation site
- Submit Feedback: Open issue/feedback page
- About/Check for Updates: Check NepTrainKit updates and display version info

## Notes

- Switching Canvas to Vispy requires a functional OpenGL environment.
- “Norm” force mode reduces data size for plotting and improves responsiveness.
- Auto loading aims to streamline repetitive workflows when working in a project directory.
- GPU backend availability depends on your platform and package build. On systems without the GPU binary or compatible drivers, NepTrainKit automatically uses the CPU backend.

