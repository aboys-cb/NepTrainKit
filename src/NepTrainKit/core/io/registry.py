#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registry for mapping result artifacts to loader implementations."""
from __future__ import annotations

import importlib
import traceback
from typing import Protocol

from loguru import logger

from NepTrainKit.core.io import ResultData, NepTrainResultData, NepDipoleResultData, NepPolarizabilityResultData
from NepTrainKit.core.utils import get_nep_type
from NepTrainKit.paths import PathLike, as_path




class ResultLoader(Protocol):
    """Loader interface used to discover and materialise result data."""

    name: str

    def matches(self, path: PathLike) -> bool:  # pragma: no cover - protocol
        """Return ``True`` when this loader can handle ``path``."""
        ...

    def load(self, path: PathLike):  # pragma: no cover - protocol
        """Materialise a result object from ``path``."""
        ...


_RESULT_LOADERS: list[ResultLoader] = []


def register_result_loader(loader: ResultLoader) -> ResultLoader:
    """Register a loader so that it participates in result discovery.

    Parameters
    ----------
    loader : ResultLoader
        An instance of a concrete `ResultLoader` subclass.

    Returns
    -------
    ResultLoader
        The same `loader` instance, allowing use as a decorator.

    Examples
    --------
    >>> from NepTrainKit.core.io.registry import DeepmdFolderLoader, NepModelTypeLoader
    >>> register_result_loader(DeepmdFolderLoader())
    >>> register_result_loader(
    ...     NepModelTypeLoader("nep_train", {0, 3},
    ...                        'NepTrainKit.core.io:NepTrainResultData')
    ... )
    >>> register_result_loader(
    ...     NepModelTypeLoader("nep_dipole", {1},
    ...                        'NepTrainKit.core.io:NepDipoleResultData')
    ... )
    >>> register_result_loader(
    ...     NepModelTypeLoader("nep_polar", {2},
    ...                        'NepTrainKit.core.io:NepPolarizabilityResultData')
    ... )
    >>> register_result_loader(OtherLoader())
    """

    _RESULT_LOADERS.append(loader)
    return loader


def matches_result_loader(path: PathLike) -> bool:
    """
       Return ``True`` if any registered loader recognises ``path``.

       Parameters
       ----------
       path : str or os.PathLike
           File or directory to be examined.

       Returns
       -------
       bool
           ``True`` if *path* is recognised by at least one loader.

       Examples
       --------
       >>> matches_result_loader("./train.xyz")
       True
       """

    candidate = as_path(path)
    for loader in _RESULT_LOADERS:
        try:
            if loader.matches(candidate):
                return True
        except Exception:  # pragma: no cover - defensive
            continue
    return False


def load_result_data(path: PathLike)->"ResultData"|None:
    """
        Load result data for *path* via the first matching loader.

        Parameters
        ----------
        path : PathLike
            File or directory to be loaded.

        Returns
        -------
        ResultData or None
            The loaded dataset if any loader recognises *path*, else ``None``.

        Examples
        --------
        >>> dataset = load_result_data("./train.xyz")
        >>> dataset.load()
        >>> print(dataset)
        """

    candidate = as_path(path)
    for loader in _RESULT_LOADERS:
        try:
            if loader.matches(candidate):
                return loader.load(candidate)
        except Exception:  # pragma: no cover - defensive
            print(traceback.format_exc())
            logger.debug(f"{ loader.name} failed to load {candidate}")
            continue
    return None


class DeepmdFolderLoader(ResultLoader):
    """Loader for DeepMD training folders."""

    name = "deepmd_folder"

    def matches(self, path: PathLike) -> bool:
        """Return ``True`` if ``path`` contains a DeepMD training directory."""
        candidate = as_path(path)
        if not candidate.is_dir():
            return False
        try:
            mod = importlib.import_module('.deepmd', __package__)
            return mod.is_deepmd_path(str(candidate))
        except Exception:
            return False

    def load(self, path: PathLike):
        """Instantiate :class:`DeepmdResultData` for ``path``."""
        mod = importlib.import_module('.deepmd', __package__)
        return mod.DeepmdResultData.from_path(str(as_path(path)))


class NepModelTypeLoader(ResultLoader):
    """Loader that selects NEP result data based on associated model type."""

    def __init__(self, name: str, model_types: set[int], factory: NepTrainResultData|NepDipoleResultData|NepPolarizabilityResultData):
        """Bind a NEP model-type loader to ``model_types`` and ``factory_path``."""
        self.name = name
        self._types = set(model_types)

        self._factory = factory
        self.model_type: int | None = None

    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` is an XYZ file associated with target NEP types."""
        candidate = as_path(path)
        if candidate.is_dir():
            return False
        dir_path = candidate.parent
        self.model_type = get_nep_type(dir_path / 'nep.txt')
        return self.model_type in self._types and candidate.suffix.lower() == '.xyz'

    def load(self, path: PathLike,*args, **kwargs):
        """Materialise the configured NEP result loader for ``path``."""

        candidate = as_path(path)
        if self._factory is  NepTrainResultData :
            return self._factory.from_path(str(candidate), model_type=self.model_type)
        return self._factory.from_path(str(candidate))


class OtherLoader(ResultLoader):
    """Fallback loader that delegates to registered importers."""

    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when a registered importer can parse ``path``."""
        candidate = as_path(path)
        try:
            imp_mod = importlib.import_module('.importers', __package__)
            return imp_mod.is_parseable(candidate)
        except Exception:
            return False

    def load(self, path: PathLike):
        """Load NEP results for ``path`` and prompt for importer options if needed."""
        candidate = as_path(path)

        inst = NepTrainResultData.from_path(str(candidate))

        imp_mod = importlib.import_module('.importers', __package__)
        lmp_imp = getattr(imp_mod, 'LammpsDumpImporter', None)
        if lmp_imp is not None and lmp_imp().matches(candidate):
            from PySide6.QtWidgets import QInputDialog

            prompt = (
                "Please enter a list of elements (corresponding to type 1..N), separated by commas or spaces. \\n"
                "For example: Si O or Si,O"
            )
            text, ok = QInputDialog.getText(None, "Element Mapping", prompt)

            if ok and text:
                raw = [t.strip() for t in str(text).replace(',', ' ').split() if t.strip()]
                if raw:
                    element_map = {i + 1: raw[i] for i in range(len(raw))}
                    existing = getattr(inst, '_import_options', {})
                    setattr(inst, '_import_options', {**existing, 'element_map': element_map})
            else:
                return None

        return inst


register_result_loader(DeepmdFolderLoader())
register_result_loader(NepModelTypeLoader("nep_train", {0, 3}, NepTrainResultData))
register_result_loader(NepModelTypeLoader("nep_dipole", {1}, NepDipoleResultData))
register_result_loader(NepModelTypeLoader("nep_polar", {2}, NepPolarizabilityResultData))
register_result_loader(OtherLoader())
