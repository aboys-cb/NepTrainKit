"""Helper utilities for sparse sampling workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional


from NepTrainKit.core import MessageManager
from NepTrainKit.core.utils import read_nep_out_file, aggregate_per_atom_to_structure
from NepTrainKit.paths import as_path
from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.structure import Structure
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - import used for type hints only
    from .base import ResultData

def pca(X: npt.NDArray[np.float32], n_components: Optional[int] = None) -> npt.NDArray[np.float32]:
    """Project a feature matrix onto its leading principal components.


    Parameters
    ----------
    X : numpy.ndarray
        Two-dimensional array containing observations by row and features by column.
    n_components : int, optional
        Number of principal components to retain. ``None`` keeps all components.

    Returns
    -------
    numpy.ndarray
        Projection of ``X`` with shape ``(n_samples, n_components)`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If ``X`` is not two dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(12, dtype=np.float32).reshape(4, 3)
    >>> pca(data, n_components=2).shape
    (4, 2)
    """
    if X.ndim != 2:
        raise ValueError('pca expects a two-dimensional array')
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    # X_centered = X
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    if n_components is None:
        n_components = n_features
    elif n_components > n_features:
        n_components = n_features
    X_pca = np.dot(X_centered, eigenvectors[:, :n_components])
    return X_pca.astype(np.float32)

def numpy_cdist(X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute pairwise Euclidean distances using broadcasting.

    Parameters
    ----------
    X : numpy.ndarray
        Array of shape ``(m, d)``.
    Y : numpy.ndarray
        Array of shape ``(n, d)``.

    Returns
    -------
    numpy.ndarray
        Distance matrix of shape ``(m, n)`` where entry ``(i, j)`` is the
        Euclidean distance between ``X[i]`` and ``Y[j]``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.zeros((2, 3), dtype=np.float32)
    >>> Y = np.ones((3, 3), dtype=np.float32)
    >>> numpy_cdist(X, Y).shape
    (2, 3)
    """
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    squared_dist = np.sum(np.square(diff), axis=2)
    return np.sqrt(squared_dist)


def farthest_point_sampling(points, n_samples, min_dist=0.1, selected_data=None) -> list[int]:
    """Greedy FPS with optional warm-start and minimum-distance constraint.

    Parameters
    ----------
    points : numpy.ndarray
        Input point set of shape ``(N, D)``.
    n_samples : int
        Maximum number of samples to select.
    min_dist : float, default=0.1
        Minimum allowed distance to any already selected point.
    selected_data : numpy.ndarray or None, optional
        Warm-start set with shape ``(M, D)``. If provided, selection respects
        the minimum distance from this set.

    Returns
    -------
    list[int]
        Indices of selected points.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.random.rand(100, 3).astype(np.float32)
    >>> idx = farthest_point_sampling(P, 5, min_dist=0.0)
    >>> len(idx) <= 5
    True
    """
    n_points = points.shape[0]

    if isinstance(selected_data, np.ndarray) and selected_data.size == 0:
        selected_data = None

    sampled_indices: list[int] = []

    if selected_data is not None:
        distances_to_samples = numpy_cdist(points, selected_data)
        min_distances = np.min(distances_to_samples, axis=1)

    else:
        first_index = 0
        sampled_indices.append(first_index)
        min_distances = np.linalg.norm(points - points[first_index], axis=1)

    while len(sampled_indices) < n_samples:
        current_index = int(np.argmax(min_distances))
        if min_distances[current_index] < float(min_dist):
            break
        sampled_indices.append(int(current_index))
        new_point = points[current_index]
        new_distances = np.linalg.norm(points - new_point, axis=1)
        min_distances = np.minimum(min_distances, new_distances)
    return sampled_indices




class SparseSampler:
    """Encapsulate descriptor preparation and sparse sampling strategies."""

    def __init__(self, result: "ResultData") -> None:
        self._result = result




    def sparse_point_selection(
        self,
        n_samples: int,
        distance: float,
        descriptor_source: str = "reduced",
        restrict_to_selection: bool = False,
        training_path: str | None = None,
    ) -> tuple[list[int], bool]:
        """Return structure indices selected by sparse sampling strategies.

        Parameters
        ----------
        n_samples : int
            Number of structures to select.
        distance : float
            Minimum feature-space distance enforced by FPS.
        descriptor_source : str, optional
            ``"reduced"`` uses PCA descriptors, ``"raw"`` uses raw descriptors.
        restrict_to_selection : bool, optional
            When ``True`` limit sampling to currently selected structures.
        training_path : str or None, optional
            Optional path to an external training dataset (XYZ file or directory)
            that seeds the distance-to-training computation in advanced mode.
        """
        # Validate descriptor availability
        dataset = getattr(self._result, "descriptor", None)
        if dataset is None or dataset.now_data.size == 0:
            MessageManager.send_message_box("No descriptor data available", "Error")
            return [], False

        # Build the base mask in "now" space, optionally restricting to selection
        reverse = False
        struct_ids_now = dataset.group_array.now_data
        mask_now = np.ones(struct_ids_now.shape[0], dtype=bool)
        if restrict_to_selection:
            sel = np.asarray(list(self._result.select_index), dtype=np.int64)
            if sel.size == 0:
                MessageManager.send_info_message("No selection found; FPS will run on full data.")
            else:
                sel_mask = np.isin(struct_ids_now, sel)
                if not np.any(sel_mask):
                    MessageManager.send_info_message(
                        "Current selection has no points on this plot; FPS will run on full data."
                    )
                else:
                    mask_now = sel_mask
                    reverse = True
                    MessageManager.send_info_message(
                        "When FPS sampling is performed in the designated area, the program will automatically deselect it, just click to delete!"
                    )

        # Collect current descriptors according to source
        # reduced -> use PCA descriptors prepared in dataset
        # raw     -> use pre-PCA descriptors cached on ResultData
        desc_now_reduced = dataset.now_data.astype(np.float32, copy=False)
        raw_all = np.asarray(getattr(self._result, "_descriptor_raw_all", np.array([], dtype=np.float32)), dtype=np.float32)

        # Align raw descriptors to now-space row order if available
        if raw_all.size != 0:
            try:
                raw_now = raw_all[dataset.data.now_indices]
            except Exception:
                raw_now = np.array([], dtype=np.float32)
        else:
            raw_now = np.array([], dtype=np.float32)

        # Optionally load/compute training descriptors
        selected_data: Optional[npt.NDArray[np.float32]] = None
        if training_path:
            try:
                t_path = as_path(training_path)
                # Try to read training structures for aggregation and fallback compute
                t_structs: list[Structure] = import_structures(t_path)
                t_counts = np.array([len(s) for s in t_structs], dtype=int) if t_structs else np.array([], dtype=int)
                # Resolve likely descriptor file next to training path
                stem = t_path.stem
                if stem == "train":
                    t_desc_path = t_path.with_name("descriptor.out")
                else:
                    t_desc_path = t_path.with_name(f"descriptor_{stem}.out")
                t_desc = read_nep_out_file(t_desc_path, dtype=np.float32, ndmin=2)
                if t_desc.size == 0 and t_structs:
                    # Compute if file missing
                    t_desc = self._result.nep_calc.get_structures_descriptor(t_structs,True)
                # Aggregate per-atom to per-structure if needed
                if t_desc.size != 0 and t_structs:
                    if t_desc.shape[0] == int(np.sum(t_counts)):
                        t_desc = aggregate_per_atom_to_structure(t_desc, t_counts, map_func=np.mean, axis=0)
                    elif t_desc.shape[0] == t_counts.shape[0]:
                        pass
                    else:
                        # Shape mismatch; best-effort fallback to compute
                        t_desc = self._result.nep_calc.get_structures_descriptor(t_structs,True)
                        if t_desc.size != 0:
                            t_desc = aggregate_per_atom_to_structure(t_desc, t_counts, map_func=np.mean, axis=0)
                selected_data = np.asarray(t_desc, dtype=np.float32) if (t_desc is not None and t_desc.size != 0) else None
            except Exception:
                # Gracefully ignore training seeding on errors
                selected_data = None

        # Prepare sampling points and optional selected_data in the same feature space
        if descriptor_source == "raw":
            # Use raw descriptor space
            if raw_now.size == 0:
                MessageManager.send_info_message("Raw descriptors not cached; falling back to reduced space.")
                points_now = desc_now_reduced
                # If training provided but points are reduced, drop seeding or reduce training to 2D below
                if selected_data is not None and selected_data.size != 0 and selected_data.shape[1] != points_now.shape[1]:
                    # Reduce combined to 2D
                    subset = points_now[mask_now]
                    try:
                        cat = np.vstack([subset.astype(np.float32, copy=False), selected_data.astype(np.float32, copy=False)])
                        reduced = pca(cat.astype(np.float32, copy=False), 2)
                        selected_data = reduced[subset.shape[0]:]
                        points_effective = reduced[:subset.shape[0]]
                    except Exception:
                        points_effective = subset
                        selected_data = None
                else:
                    points_effective = points_now[mask_now]
            else:
                points_now = raw_now
                points_effective = points_now[mask_now]
                # Ensure selected_data matches dimensionality if provided
                if selected_data is not None and selected_data.size != 0 and selected_data.shape[1] != points_now.shape[1]:
                    # Reduce raw current subset + training to 2D
                    subset = points_effective
                    try:
                        cat = np.vstack([subset.astype(np.float32, copy=False), selected_data.astype(np.float32, copy=False)])
                        reduced = pca(cat.astype(np.float32, copy=False), 2)
                        selected_data = reduced[subset.shape[0]:]
                        points_effective = reduced[:subset.shape[0]]
                    except Exception:
                        # On failure, just drop training seed to keep going in raw space
                        selected_data = None
        else:
            # Use reduced (PCA) space
            if selected_data is not None and selected_data.size != 0:
                # Reduce after merging raw current subset with training seed for consistent space
                if raw_now.size == 0:
                    # If raw not available, reduce using current reduced space only
                    subset = desc_now_reduced[mask_now]
                    # Try to bring training into the same 2D space by joint PCA with subset
                    try:
                        cat = np.vstack([subset.astype(np.float32, copy=False), selected_data.astype(np.float32, copy=False)])
                        reduced = pca(cat.astype(np.float32, copy=False), 2)
                        points_effective = reduced[:subset.shape[0]]
                        selected_data = reduced[subset.shape[0]:]
                    except Exception:
                        points_effective = subset
                        selected_data = None
                else:
                    subset_raw = raw_now[mask_now]
                    try:
                        cat = np.vstack([subset_raw.astype(np.float32, copy=False), selected_data.astype(np.float32, copy=False)])
                        reduced = pca(cat.astype(np.float32, copy=False), 2)
                        points_effective = reduced[:subset_raw.shape[0]]
                        selected_data = reduced[subset_raw.shape[0]:]
                    except Exception:
                        # Fall back to existing reduced now-data
                        points_effective = desc_now_reduced[mask_now]
                        selected_data = None
            else:
                points_effective = desc_now_reduced[mask_now]

        # Run FPS on the prepared subset
        if points_effective.size == 0:
            global_rows = np.array([], dtype=np.int64)
        else:
            idx_local = farthest_point_sampling(points_effective, n_samples=n_samples, min_dist=distance, selected_data=selected_data)
            if len(idx_local) == 0:
                global_rows = np.array([], dtype=np.int64)
            else:
                rows_now = np.where(mask_now)[0]
                global_rows = rows_now[np.asarray(idx_local, dtype=np.int64)]

        structures = dataset.group_array.now_data[global_rows]
        return structures.tolist(), reverse
