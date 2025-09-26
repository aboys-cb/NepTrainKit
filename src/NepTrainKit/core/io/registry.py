#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registry for mapping result artifacts to loader implementations."""
from __future__ import annotations

import importlib
import traceback
from typing import Protocol

from loguru import logger

from NepTrainKit.core.io.utils import get_nep_type
from NepTrainKit.paths import PathLike, as_path


class ResultDataProtocol(Protocol):
    """Protocol describing the subset of result data objects we rely on."""

    load_flag: bool

    @classmethod
    def from_path(cls, path: PathLike, *args, **kwargs):  # pragma: no cover - protocol
        """Create an instance from ``path`` using implementation-specific logic."""
        ...


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
    """Register ``loader`` so it participates in result discovery."""

    _RESULT_LOADERS.append(loader)
    return loader


def matches_result_loader(path: PathLike) -> bool:
    """Return ``True`` if any registered loader recognises ``path``."""

    candidate = as_path(path)
    for loader in _RESULT_LOADERS:
        try:
            if loader.matches(candidate):
                return True
        except Exception:  # pragma: no cover - defensive
            continue
    return False


def load_result_data(path: PathLike):
    """Load result data for ``path`` via the first matching loader."""

    candidate = as_path(path)
    for loader in _RESULT_LOADERS:
        try:
            if loader.matches(candidate):
                return loader.load(candidate)
        except Exception:  # pragma: no cover - defensive
            logger.debug("%s failed to load %s", loader.name, candidate)
            continue
    return None


class DeepmdFolderLoader:
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


class NepModelTypeLoader:
    """Loader that selects NEP result data based on associated model type."""

    def __init__(self, name: str, model_types: set[int], factory_path: str):
        """Bind a NEP model-type loader to ``model_types`` and ``factory_path``."""
        self.name = name
        self._types = set(model_types)
        self._factory_path = factory_path
        self._factory = None
        self.model_type: int | None = None

    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` is an XYZ file associated with target NEP types."""
        candidate = as_path(path)
        if candidate.is_dir():
            return False
        dir_path = candidate.parent
        self.model_type = get_nep_type(dir_path / 'nep.txt')
        return self.model_type in self._types and candidate.suffix.lower() == '.xyz'

    def load(self, path: PathLike):
        """Materialise the configured NEP result loader for ``path``."""
        if self._factory is None:
            module_name, cls_name = self._factory_path.split(':', 1)
            mod = importlib.import_module(module_name)
            self._factory = getattr(mod, cls_name)
        loader = self._factory
        candidate = as_path(path)
        if self._factory_path.endswith(':NepTrainResultData'):
            return loader.from_path(str(candidate), model_type=self.model_type)
        return loader.from_path(str(candidate))


class OtherLoader:
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
        nep_mod = importlib.import_module('.nep', __package__)
        inst = nep_mod.NepTrainResultData.from_path(str(candidate))

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
register_result_loader(NepModelTypeLoader("nep_train", {0, 3}, 'NepTrainKit.core.io.nep:NepTrainResultData'))
register_result_loader(NepModelTypeLoader("nep_dipole", {1}, 'NepTrainKit.core.io.nep:NepDipoleResultData'))
register_result_loader(NepModelTypeLoader("nep_polar", {2}, 'NepTrainKit.core.io.nep:NepPolarizabilityResultData'))
register_result_loader(OtherLoader())
