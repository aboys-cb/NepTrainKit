"""Helper utilities for sparse sampling workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

try:
    from NepTrainKit._native import _sampling as _native_sampling
except ImportError:  # Source trees may not have built native helpers yet.
    _native_sampling = None

from NepTrainKit.core import MessageManager
from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.utils import aggregate_per_atom_to_structure, read_nep_out_file
from NepTrainKit.paths import as_path

from .sampling_features import build_sampling_feature_blocks
from .sampling_plan import (
    PhysicsSamplingPlan,
    build_physics_sampling_plan,
    build_result_physics_sampling_plan,
)
from .sampling_recommendation import (
    PhysicsSamplingRecommendation,
    SamplingPolicy,
    recommend_physics_sampling_from_blocks,
    select_physics_budget,
)

if TYPE_CHECKING:  # pragma: no cover - import used for type hints only
    from .base import ResultData

def pca(
    X: npt.NDArray[np.float32],
    n_components: int | None = None,
) -> npt.NDArray[np.float32]:
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


def _farthest_point_sampling_numpy(
    points,
    n_samples,
    min_dist=0.1,
    selected_data=None,
) -> list[int]:
    """NumPy reference for greedy FPS."""
    points = np.asarray(points)
    n_points = points.shape[0]
    if n_points == 0 or int(n_samples) <= 0:
        return []

    if isinstance(selected_data, np.ndarray) and selected_data.size == 0:
        selected_data = None

    sampled_indices: list[int] = []
    selected_mask = np.zeros(n_points, dtype=bool)

    if selected_data is not None:
        distances_to_samples = numpy_cdist(points, selected_data)
        min_distances = np.min(distances_to_samples, axis=1)

    else:
        first_index = 0
        sampled_indices.append(first_index)
        selected_mask[first_index] = True
        min_distances = np.linalg.norm(points - points[first_index], axis=1)
        min_distances[selected_mask] = -np.inf

    max_samples = min(int(n_samples), n_points)
    while len(sampled_indices) < max_samples:
        current_index = int(np.argmax(min_distances))
        if selected_mask[current_index] or not np.isfinite(min_distances[current_index]):
            break
        if min_distances[current_index] < float(min_dist):
            break
        sampled_indices.append(int(current_index))
        selected_mask[current_index] = True
        new_point = points[current_index]
        new_distances = np.linalg.norm(points - new_point, axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected_mask] = -np.inf
    return sampled_indices


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
    points_array = np.asarray(points)
    n_points = points_array.shape[0]
    if n_points == 0 or int(n_samples) <= 0:
        return []

    if isinstance(selected_data, np.ndarray) and selected_data.size == 0:
        selected_data = None
    selected_array = None if selected_data is None else np.asarray(selected_data)
    if _native_sampling is not None and hasattr(
        _native_sampling,
        "farthest_point_sampling",
    ):
        return list(
            _native_sampling.farthest_point_sampling(
                points_array,
                int(n_samples),
                float(min_dist),
                selected_array,
            )
        )
    return _farthest_point_sampling_numpy(
        points_array,
        n_samples=int(n_samples),
        min_dist=float(min_dist),
        selected_data=selected_array,
    )


def allocate_sqrt_quotas(
    group_sizes: dict[Hashable, int],
    n_samples: int,
) -> dict[Hashable, int]:
    """Allocate one slot per non-empty group, then distribute by sqrt(size)."""
    sizes = {key: int(size) for key, size in group_sizes.items() if int(size) > 0}
    if not sizes or int(n_samples) <= 0:
        return {}
    budget = min(int(n_samples), sum(sizes.values()))
    if budget < len(sizes):
        raise ValueError(
            f"Target count {budget} is smaller than the {len(sizes)} element-set groups. "
            "Increase the target count or remove unneeded systems."
        )

    quotas = {key: 1 for key in sizes}
    remaining = budget - len(quotas)
    while remaining > 0:
        eligible = [key for key in sizes if quotas[key] < sizes[key]]
        if not eligible:
            break
        weight_sum = sum(sqrt(sizes[key]) for key in eligible)
        raw = {key: remaining * sqrt(sizes[key]) / weight_sum for key in eligible}
        granted = 0
        for key in eligible:
            addition = min(sizes[key] - quotas[key], int(np.floor(raw[key])))
            quotas[key] += addition
            remaining -= addition
            granted += addition
        if remaining <= 0:
            break
        order = sorted(
            eligible,
            key=lambda key: (-(raw[key] - np.floor(raw[key])), -sizes[key], repr(key)),
        )
        remainder_granted = 0
        for key in order:
            if remaining <= 0:
                break
            if quotas[key] < sizes[key]:
                quotas[key] += 1
                remaining -= 1
                remainder_granted += 1
        if granted == 0 and remainder_granted == 0:
            break
    return quotas


def centered_fps(
    points,
    n_samples: int,
    min_dist: float,
    selected_data=None,
) -> list[int]:
    """Run FPS from the feature-space center, or from a warm-start set."""
    points = np.asarray(points)
    if points.shape[0] == 0 or int(n_samples) <= 0:
        return []
    if selected_data is not None and np.asarray(selected_data).size > 0:
        return farthest_point_sampling(
            points,
            n_samples=int(n_samples),
            min_dist=float(min_dist),
            selected_data=np.asarray(selected_data),
        )

    center = np.mean(points, axis=0)
    center_index = int(np.argmin(np.linalg.norm(points - center, axis=1)))
    order = np.concatenate(
        (np.asarray([center_index], dtype=int), np.delete(np.arange(points.shape[0]), center_index))
    )
    local_indices = farthest_point_sampling(
        points[order],
        n_samples=int(n_samples),
        min_dist=float(min_dist),
    )
    return order[np.asarray(local_indices, dtype=int)].tolist()


def structure_element_set_key(structure) -> tuple[str, ...]:
    """Return a stable element-set key for ASE or NepTrainKit structures."""
    getter = getattr(structure, "get_chemical_symbols", None)
    if callable(getter):
        symbols = getter()
    else:
        symbols = getattr(structure, "elements", ())
    return tuple(sorted({str(symbol) for symbol in symbols}))


def incremental_fps_with_r2(
    points: npt.NDArray[np.float32],
    r2_threshold: float,
    n_samples: int | None = None,
    min_dist: float = 0.1,
    selected_data: npt.NDArray[np.float32] | None = None,
) -> tuple[list[int], float]:
    """FPS that stops once selected centers cover enough descriptor variance.

    Parameters
    ----------
    points : numpy.ndarray
        Candidate point set of shape ``(N, D)``.
    r2_threshold : float
        Target coverage R². Sampling stops when nearest-center residual variance
        falls below ``1 - r2_threshold`` of total descriptor variance.
    n_samples : int or None, optional
        Maximum number of samples to draw; ``None`` or ``<=0`` defaults to ``N``.
    min_dist : float, default=0.1
        Minimum allowed distance to any already selected point.
    selected_data : numpy.ndarray or None, optional
        Warm-start centers used by both FPS distances and coverage R².

    Returns
    -------
    (list[int], float)
        Selected indices and the final R² value.
    """
    n_points = int(points.shape[0])
    if n_points == 0:
        return [], 0.0
    if n_samples is None or n_samples <= 0 or n_samples > n_points:
        n_samples = n_points

    points = np.asarray(points, dtype=np.float32)
    overall_mean = np.mean(points, axis=0)
    total_variance = float(np.sum((points - overall_mean) ** 2))

    # Initialise distance field, optionally warm-started
    sampled_indices: list[int] = []
    selected_mask = np.zeros(n_points, dtype=bool)
    if isinstance(selected_data, np.ndarray) and selected_data.size != 0:
        distances_to_samples = numpy_cdist(points, selected_data)
        min_distances = np.min(distances_to_samples, axis=1)
    else:
        first_index = 0
        sampled_indices.append(first_index)
        selected_mask[first_index] = True
        min_distances = np.linalg.norm(points - points[first_index], axis=1)
        min_distances[selected_mask] = -np.inf

    def _current_r2() -> float:
        if total_variance <= 0.0:
            return 1.0
        residual_distances = np.where(
            np.isfinite(min_distances),
            np.maximum(min_distances, 0.0),
            0.0,
        )
        residual_variance = float(np.sum(np.square(residual_distances)))
        return float(np.clip(1.0 - residual_variance / total_variance, 0.0, 1.0))

    # Degenerate candidates need one representative only when no warm start exists.
    if total_variance <= 0.0:
        return sampled_indices, 1.0

    r2 = _current_r2()
    if r2 >= r2_threshold or len(sampled_indices) >= n_samples:
        return sampled_indices, r2

    while len(sampled_indices) < n_samples:
        current_index = int(np.argmax(min_distances))
        if selected_mask[current_index] or not np.isfinite(min_distances[current_index]):
            break
        if min_distances[current_index] < float(min_dist):
            break
        sampled_indices.append(current_index)
        selected_mask[current_index] = True
        new_point = points[current_index]
        new_distances = np.linalg.norm(points - new_point, axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected_mask] = -np.inf
        r2 = _current_r2()
        if r2 >= r2_threshold:
            break

    return sampled_indices, r2




class SparseSampler:
    """Encapsulate descriptor preparation and sparse sampling strategies."""

    def __init__(self, result: "ResultData") -> None:
        self._result = result
        self._physics_plan_cache_key: tuple[bool, bytes] | None = None
        self._physics_plan_cache: PhysicsSamplingPlan | None = None
        self._atomic_descriptor_cache_key: bytes | None = None
        self._atomic_descriptor_cache = np.array([], dtype=np.float32)

    def _candidate_atomic_descriptors(
        self,
        structures,
        structure_ids,
    ) -> np.ndarray | None:
        """Return per-atom descriptors aligned with candidate structures."""
        ids = np.ascontiguousarray(structure_ids, dtype=np.int64)
        all_atomic = np.asarray(
            getattr(self._result, "_descriptor_atom_all", np.array([])),
            dtype=np.float32,
        )
        atom_counts = np.asarray(getattr(self._result, "atoms_num_list", ()), dtype=int)
        if all_atomic.size and atom_counts.size and all_atomic.shape[0] == int(np.sum(atom_counts)):
            offsets = np.concatenate(([0], np.cumsum(atom_counts)))
            return np.ascontiguousarray(
                np.vstack(
                    [all_atomic[offsets[index] : offsets[index + 1]] for index in ids]
                ),
                dtype=np.float32,
            )
        key = ids.tobytes()
        if key == self._atomic_descriptor_cache_key and self._atomic_descriptor_cache.size:
            return self._atomic_descriptor_cache
        calculator = getattr(self._result, "nep_calc", None)
        if calculator is None:
            return None
        values = np.ascontiguousarray(
            calculator.descriptors(structures, mean=False),
            dtype=np.float32,
        )
        self._atomic_descriptor_cache_key = key
        self._atomic_descriptor_cache = values
        return values

    def _physics_plan_for_candidates(
        self,
        structures,
        structure_ids,
        *,
        spin_model: bool,
    ) -> PhysicsSamplingPlan:
        ids = np.ascontiguousarray(structure_ids, dtype=np.int64)
        cache_key = (bool(spin_model), ids.tobytes())
        if (
            cache_key == self._physics_plan_cache_key
            and self._physics_plan_cache is not None
        ):
            return self._physics_plan_cache
        plan = None
        if (
            getattr(self._result, "data_xyz_path", None) is not None
            and getattr(self._result, "descriptor_path", None) is not None
        ):
            plan = build_result_physics_sampling_plan(
                self._result,
                ids,
                spin_model=spin_model,
            )
        if plan is None:
            plan = build_physics_sampling_plan(
                structures,
                spin_model=spin_model,
                source_indices=ids,
            )
        self._physics_plan_cache_key = cache_key
        self._physics_plan_cache = plan
        return plan

    def _load_training_context(
        self,
        training_path: str | None,
        *,
        strategy: str,
        spin_model: bool,
    ) -> tuple[
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
        list[Structure],
        PhysicsSamplingPlan | None,
    ]:
        """Load one optional training seed in the raw descriptor space."""
        if not training_path:
            return None, None, [], None
        t_path = as_path(training_path)
        training_structures = list(import_structures(t_path))
        atom_counts = (
            np.asarray([len(structure) for structure in training_structures], dtype=int)
            if training_structures
            else np.asarray([], dtype=int)
        )
        stem = t_path.stem
        descriptor_path = (
            t_path.with_name("descriptor.out")
            if stem == "train"
            else t_path.with_name(f"descriptor_{stem}.out")
        )
        descriptors = read_nep_out_file(
            descriptor_path,
            dtype=np.float32,
            ndmin=2,
        )
        if descriptors.size == 0 and training_structures:
            descriptors = self._result.nep_calc.descriptors(
                training_structures,
                mean=True,
            )
        atomic_descriptors = None
        if descriptors.size != 0 and training_structures:
            if descriptors.shape[0] == int(np.sum(atom_counts)):
                atomic_descriptors = np.asarray(descriptors, dtype=np.float32)
                descriptors = aggregate_per_atom_to_structure(
                    descriptors,
                    atom_counts,
                    map_func=np.mean,
                    axis=0,
                )
            elif descriptors.shape[0] != atom_counts.shape[0]:
                descriptors = self._result.nep_calc.descriptors(
                    training_structures,
                    mean=True,
                )
        selected_data = (
            np.asarray(descriptors, dtype=np.float32)
            if descriptors is not None and descriptors.size != 0
            else None
        )
        if strategy in {"element_set", "physics"} and (
            not training_structures
            or selected_data is None
            or selected_data.shape[0] != len(training_structures)
        ):
            raise ValueError("training structures and descriptors are not aligned")
        calculator = getattr(self._result, "nep_calc", None)
        if (
            strategy == "physics"
            and training_structures
            and atomic_descriptors is None
            and calculator is not None
        ):
            atomic_descriptors = np.asarray(
                calculator.descriptors(training_structures, mean=False),
                dtype=np.float32,
            )

        training_plan = None
        if strategy == "physics":
            training_plan = build_physics_sampling_plan(
                training_structures,
                spin_model=spin_model,
            )
            if spin_model and training_plan.missing_spin_indices:
                first = training_plan.missing_spin_indices[0] + 1
                raise ValueError(
                    "the detected spin model requires canonical spin:R:3 "
                    "in the existing training set; "
                    f"first missing structure: {first}"
                )
        return selected_data, atomic_descriptors, training_structures, training_plan

    def recommend_physics_sample_count(
        self,
        *,
        restrict_to_selection: bool = False,
        training_path: str | None = None,
        policy: SamplingPolicy = "balanced",
    ) -> PhysicsSamplingRecommendation:
        """Recommend a physics-aware FPS count without changing the selection."""
        dataset = getattr(self._result, "descriptor", None)
        if dataset is None or dataset.now_data.size == 0:
            raise ValueError("No descriptor data available")

        structure_ids = np.asarray(dataset.group_array.now_data, dtype=np.int64)
        mask = np.ones(structure_ids.shape[0], dtype=bool)
        if restrict_to_selection:
            selected = np.asarray(list(self._result.select_index), dtype=np.int64)
            selected_mask = np.isin(structure_ids, selected)
            if selected.size and np.any(selected_mask):
                mask = selected_mask

        raw_all = np.asarray(
            getattr(
                self._result,
                "_descriptor_raw_all",
                np.asarray([], dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if raw_all.size == 0:
            raise ValueError(
                "Raw structure descriptors are required for physics-aware recommendations."
            )
        try:
            raw_now = raw_all[dataset.data.now_indices]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "Raw structure descriptors do not align with the active dataset."
            ) from exc

        rows = np.where(mask)[0]
        candidate_ids = structure_ids[rows]
        candidate_structures = [
            self._result.structure.all_data[int(structure_id)]
            for structure_id in candidate_ids
        ]
        spin_model = bool(getattr(self._result, "is_spin_model", False))
        if not spin_model:
            spin_model = bool(
                getattr(getattr(self._result, "nep_calc", None), "is_spin_model", False)
            )
        plan = self._physics_plan_for_candidates(
            candidate_structures,
            candidate_ids,
            spin_model=spin_model,
        )
        if spin_model and plan.missing_spin_indices:
            first = plan.missing_spin_indices[0] + 1
            raise ValueError(
                "the detected spin model requires canonical spin:R:3 "
                f"in every candidate; first missing structure: {first}"
            )

        (
            existing_descriptors,
            existing_atomic_descriptors,
            training_structures,
            existing_plan,
        ) = (
            self._load_training_context(
                training_path,
                strategy="physics",
                spin_model=spin_model,
            )
        )
        if (
            existing_descriptors is not None
            and existing_descriptors.shape[1] != raw_now.shape[1]
        ):
            raise ValueError(
                "Existing training descriptors do not match the loaded raw descriptor dimensions."
            )
        candidate_atomic_descriptors = self._candidate_atomic_descriptors(
            candidate_structures,
            candidate_ids,
        )
        if existing_descriptors is not None and (
            candidate_atomic_descriptors is None
            or existing_atomic_descriptors is None
        ):
            candidate_atomic_descriptors = None
            existing_atomic_descriptors = None
        candidate_blocks = build_sampling_feature_blocks(
            candidate_structures,
            raw_now[rows],
            per_atom_descriptors=candidate_atomic_descriptors,
            spin_model=spin_model,
        )
        existing_blocks = (
            None
            if existing_descriptors is None
            else build_sampling_feature_blocks(
                training_structures,
                existing_descriptors,
                per_atom_descriptors=existing_atomic_descriptors,
                spin_model=spin_model,
            )
        )
        return recommend_physics_sampling_from_blocks(
            candidate_blocks,
            plan,
            existing=existing_blocks,
            existing_plan=existing_plan,
            policy=policy,
        )




    def sparse_point_selection(
        self,
        n_samples: int,
        distance: float,
        descriptor_source: str = "reduced",
        restrict_to_selection: bool = False,
        training_path: str | None = None,
        sampling_mode: str = "count",
        r2_threshold: float = 0.9,
        selection_strategy: str = "global",
        physics_count_mode: str = "limit",
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
        sampling_mode : {"count", "r2"}, optional
            ``"count"`` performs standard fixed-count FPS, ``"r2"`` stops early once the
            selected set reaches the target R² on the candidate points.
        r2_threshold : float, optional
            Target R² used when ``sampling_mode`` is ``"r2"``.
        selection_strategy : {"global", "element_set", "physics"}, optional
            ``"element_set"`` applies sqrt-size quotas and centered FPS within
            each chemical element set. ``"physics"`` balances exact
            composition, structural phase, and magnetic order for detected spin
            models. Both structured strategies require raw descriptors. Physics
            sampling supports either a user total limit or an automatic
            descriptor-coverage count.
        physics_count_mode : {"limit", "automatic"}, optional
            ``"limit"`` distributes ``n_samples`` as one global upper limit.
            ``"automatic"`` derives and selects the count independently within
            each physical stratum before summing the result.
        """
        self._result._last_sparse_group_report = {}
        self._result._last_sparse_physics_plan = None
        self._result._last_sparse_physics_recommendation = None
        self._result._last_sparse_coverage_r2 = None

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

        strategy = str(selection_strategy or "global").strip().lower()
        if strategy not in {"global", "element_set", "physics"}:
            MessageManager.send_message_box(
                f"Unsupported FPS selection strategy: {selection_strategy}",
                "Error",
            )
            return [], reverse
        count_mode = str(physics_count_mode or "limit").strip().lower()
        if count_mode not in {"limit", "automatic"}:
            MessageManager.send_message_box(
                f"Unsupported physics count mode: {physics_count_mode}",
                "Error",
            )
            return [], reverse
        requires_limit = strategy != "physics" or count_mode == "limit"
        if strategy in {"element_set", "physics"} and requires_limit and int(n_samples) <= 0:
            MessageManager.send_message_box(
                "Structured balanced FPS requires a positive sample limit.",
                "Error",
            )
            return [], reverse

        spin_model = bool(getattr(self._result, "is_spin_model", False))
        if not spin_model:
            calculator = getattr(self._result, "nep_calc", None)
            spin_model = bool(getattr(calculator, "is_spin_model", False))

        # Optionally load/compute training descriptors
        selected_data: npt.NDArray[np.float32] | None = None
        selected_atomic_data: npt.NDArray[np.float32] | None = None
        training_structures: list[Structure] = []
        training_physics_plan: PhysicsSamplingPlan | None = None
        if training_path:
            try:
                (
                    selected_data,
                    selected_atomic_data,
                    training_structures,
                    training_physics_plan,
                ) = self._load_training_context(
                    training_path,
                    strategy=strategy,
                    spin_model=spin_model,
                )
            except Exception as exc:
                if strategy in {"element_set", "physics"}:
                    MessageManager.send_message_box(
                        f"Unable to use the existing training dataset for balanced FPS: {exc}",
                        "Error",
                    )
                    return [], reverse
                # Gracefully ignore training seeding on errors
                selected_data = None
                training_structures = []

        if strategy in {"element_set", "physics"}:
            descriptor_source = "raw"
            sampling_mode = "count"
            if raw_now.size == 0:
                MessageManager.send_message_box(
                    "Raw structure descriptors are required for structured balanced FPS.",
                    "Error",
                )
                return [], reverse
            if selected_data is not None and selected_data.shape[1] != raw_now.shape[1]:
                MessageManager.send_message_box(
                    "Existing training descriptors do not match the loaded raw descriptor dimensions.",
                    "Error",
                )
                return [], reverse

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
        elif strategy in {"element_set", "physics"}:
            rows_now = np.where(mask_now)[0]
            candidate_structure_ids = struct_ids_now[rows_now]
            candidate_structures = [
                self._result.structure.all_data[int(structure_id)]
                for structure_id in candidate_structure_ids
            ]

            candidate_physics_plan: PhysicsSamplingPlan | None = None
            try:
                if strategy == "physics":
                    candidate_physics_plan = self._physics_plan_for_candidates(
                        candidate_structures,
                        candidate_structure_ids,
                        spin_model=spin_model,
                    )
                    if spin_model and candidate_physics_plan.missing_spin_indices:
                        first = candidate_physics_plan.missing_spin_indices[0] + 1
                        raise ValueError(
                            "the detected spin model requires canonical spin:R:3 "
                            f"in every candidate; first missing structure: {first}"
                        )
                    candidate_groups = candidate_physics_plan.group_indices()
                else:
                    candidate_groups = defaultdict(list)
                    for local_row, structure in enumerate(candidate_structures):
                        candidate_groups[structure_element_set_key(structure)].append(
                            local_row
                        )
                    quotas = allocate_sqrt_quotas(
                        {key: len(indices) for key, indices in candidate_groups.items()},
                        n_samples,
                    )
            except (RuntimeError, ValueError) as exc:
                MessageManager.send_message_box(str(exc), "Error")
                return [], reverse

            if strategy == "physics":
                candidate_atomic = self._candidate_atomic_descriptors(
                    candidate_structures,
                    candidate_structure_ids,
                )
                if selected_data is not None and (
                    candidate_atomic is None or selected_atomic_data is None
                ):
                    candidate_atomic = None
                    selected_atomic_data = None
                candidate_blocks = build_sampling_feature_blocks(
                    candidate_structures,
                    points_effective,
                    per_atom_descriptors=candidate_atomic,
                    spin_model=spin_model,
                )
                existing_blocks = (
                    None
                    if selected_data is None
                    else build_sampling_feature_blocks(
                        training_structures,
                        selected_data,
                        per_atom_descriptors=selected_atomic_data,
                        spin_model=spin_model,
                    )
                )
                try:
                    if count_mode == "automatic":
                        recommendation = recommend_physics_sampling_from_blocks(
                            candidate_blocks,
                            candidate_physics_plan,
                            existing=existing_blocks,
                            existing_plan=training_physics_plan,
                            policy="balanced",
                        )
                        selected_local_rows = list(recommendation.selected_indices)
                        group_results = recommendation.groups
                        self._result._last_sparse_physics_recommendation = recommendation
                    else:
                        budget_selection = select_physics_budget(
                            candidate_blocks,
                            candidate_physics_plan,
                            n_samples=n_samples,
                            existing=existing_blocks,
                            existing_plan=training_physics_plan,
                            min_distance=distance,
                        )
                        selected_local_rows = list(budget_selection.selected_indices)
                        group_results = budget_selection.groups
                except ValueError as exc:
                    MessageManager.send_message_box(str(exc), "Error")
                    return [], reverse
                group_report = {
                    group.stratum: {
                        "candidate_count": group.candidate_count,
                        "existing_count": group.existing_count,
                        "selected_count": int(
                            getattr(
                                group,
                                "selected_count",
                                getattr(group, "recommended_count", 0),
                            )
                        ),
                    }
                    for group in group_results
                }
            else:
                training_groups = defaultdict(list)
                for index, structure in enumerate(training_structures):
                    training_groups[structure_element_set_key(structure)].append(index)
                selected_local_rows = []
                group_report = {}
                for key in sorted(candidate_groups):
                    candidate_rows = candidate_groups[key]
                    warm_rows = training_groups.get(key, [])
                    warm_points = (
                        np.asarray(selected_data[warm_rows], dtype=np.float32)
                        if selected_data is not None and warm_rows
                        else None
                    )
                    chosen_rows = centered_fps(
                        points_effective[candidate_rows],
                        n_samples=quotas[key],
                        min_dist=distance,
                        selected_data=warm_points,
                    )
                    selected_local_rows.extend(
                        candidate_rows[index] for index in chosen_rows
                    )
                    group_report[key] = {
                        "candidate_count": len(candidate_rows),
                        "existing_count": len(warm_rows),
                        "selected_count": len(chosen_rows),
                    }
            self._result._last_sparse_group_report = group_report
            self._result._last_sparse_physics_plan = candidate_physics_plan
            global_rows = rows_now[np.asarray(sorted(selected_local_rows), dtype=np.int64)]
        else:
            mode = (sampling_mode or "count").lower()
            if mode == "r2":
                max_samples = n_samples if n_samples > 0 else points_effective.shape[0]
                idx_local, coverage_r2 = incremental_fps_with_r2(
                    points_effective,
                    r2_threshold=float(r2_threshold),
                    n_samples=max_samples,
                    min_dist=distance,
                    selected_data=selected_data,
                )
                self._result._last_sparse_coverage_r2 = coverage_r2
            else:
                idx_local = farthest_point_sampling(
                    points_effective,
                    n_samples=n_samples,
                    min_dist=distance,
                    selected_data=selected_data,
                )
            if len(idx_local) == 0:
                global_rows = np.array([], dtype=np.int64)
            else:
                rows_now = np.where(mask_now)[0]
                global_rows = rows_now[np.asarray(idx_local, dtype=np.int64)]

        structures = dataset.group_array.now_data[global_rows]
        return structures.tolist(), reverse
