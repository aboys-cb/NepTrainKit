# Data Management

This module provides project- and version-oriented management of NEP datasets. You can organize models under projects, track versions, tag datasets, and open their working directories.

## 1. Concepts

- **Project**: A tree of folders that groups related models. Supports nested hierarchy.
- **Model (Version)**: A dataset entry under a project. Stores metadata: path/URL, size, energy/force/virial metrics, tags, notes, created time.
- **Tags**: Colored labels to classify or filter models. Managed centrally and applied to models.

## 2. UI Overview

- Left panel: Project tree. Right panel: Model list of the selected project.
- Context menus on both panels provide creation, modification, deletion, and utility actions.

## 3. Project Operations

- New: Right‑click the project tree → New. Choose parent, name, notes.
- Modify: Right‑click an item → Modify. Edit name, notes, parent.
- Delete: Right‑click an item → Delete. Deleting a node removes all descendants.

## 4. Model (Version) Operations

- New: Right‑click the model list → New. Fill in:
  - Name, notes, model type
  - Train path (local folder or HTTP URL)
  - Energy/Force/Virial scores (optional metadata)
  - Parent model (to build a version chain)
  - Tags
- Modify/Delete: Right‑click an item → Modify/Delete.
- Open Folder: Opens the `train_path` in your OS file manager or browser (if URL).

## 5. Tags and Search

- Manage Tags: Right‑click the model list → Manage Tags to add/rename/remove tags and colors.
- Quick Search: Press `Ctrl+F` in the model list to open advanced search. Filter by project, type, text, time, metrics, and tags.

## 6. Database Location

- The app stores management data in a SQLite database under the user config directory:
  - Windows: `C:\Users\<You>\AppData\Local\NepTrainKit\mlpman.db`
  - Linux: `~/.config/NepTrainKit/mlpman.db`

## 7. Tips

- Hierarchies: Use parent/child for simple version trees.
- External sources: Set `train_path` to a Git/HTTP URL to reference online data.
- Bulk review: Use tags to triage candidate datasets before promotion.

