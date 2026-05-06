"""Registration helpers for pluggable "card" components.

This module provides a tiny registry for UI/processing "cards" that can be
discovered dynamically from a directory. It avoids import-time side effects by
loading modules on demand and expecting them to self-register.

Examples
--------
>>> from NepTrainKit.core.card_manager import CardManager
>>> @CardManager.register_card
... class MyCard: ...
... 
>>> 'MyCard' in CardManager.card_info_dict
True
"""
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass(frozen=True)
class CardContributor:
    """Public contributor metadata attached to a Make Dataset card."""

    name: str
    role: str = "author"
    email: str = ""
    url: str = ""
    affiliation: str = ""


@dataclass(frozen=True)
class CardMetadata:
    """Display metadata collected when a card class is registered."""

    class_name: str
    card_name: str
    group: str | None = None
    description: str = ""
    version: str = ""
    contributors: tuple[CardContributor, ...] = ()
    maintainer: str = ""
    license: str = ""
    citation: str = ""
    docs_url: str = ""
    source_path: str = ""


def _as_contributor(value) -> CardContributor | None:
    """Normalize a contributor declaration into :class:`CardContributor`."""
    if isinstance(value, CardContributor):
        return value
    if isinstance(value, str):
        name = value.strip()
        return CardContributor(name=name) if name else None
    if isinstance(value, dict):
        name = str(value.get("name", "")).strip()
        if not name:
            return None
        return CardContributor(
            name=name,
            role=str(value.get("role", "author") or "author").strip(),
            email=str(value.get("email", "") or "").strip(),
            url=str(value.get("url", "") or "").strip(),
            affiliation=str(value.get("affiliation", "") or "").strip(),
        )
    return None


def _normalize_contributors(raw) -> tuple[CardContributor, ...]:
    """Return a stable tuple of contributor metadata."""
    if raw is None:
        return ()
    if isinstance(raw, (str, dict, CardContributor)):
        raw = [raw]
    contributors: list[CardContributor] = []
    try:
        iterator = iter(raw)
    except TypeError:
        return ()
    for item in iterator:
        contributor = _as_contributor(item)
        if contributor is not None:
            contributors.append(contributor)
    return tuple(contributors)


def build_card_metadata(card_class) -> CardMetadata:
    """Build display metadata from a registered card class."""
    try:
        source_path = str(Path(inspect.getfile(card_class)).resolve())
    except (TypeError, OSError):
        source_path = ""

    description = str(getattr(card_class, "description", "") or "").strip()
    if not description:
        doc = inspect.getdoc(card_class) or ""
        description = doc.splitlines()[0].strip() if doc else ""

    return CardMetadata(
        class_name=card_class.__name__,
        card_name=str(getattr(card_class, "card_name", card_class.__name__) or card_class.__name__),
        group=getattr(card_class, "group", None),
        description=description,
        version=str(getattr(card_class, "card_version", "") or "").strip(),
        contributors=_normalize_contributors(getattr(card_class, "contributors", None)),
        maintainer=str(getattr(card_class, "maintainer", "") or "").strip(),
        license=str(getattr(card_class, "license", "") or "").strip(),
        citation=str(getattr(card_class, "citation", "") or "").strip(),
        docs_url=str(getattr(card_class, "docs_url", "") or "").strip(),
        source_path=source_path,
    )

class CardManager:
    """Simple registry mapping class names to card classes.

    Notes
    -----
    - Uses class name as the unique key; later registrations overwrite prior ones.
    - Intended to be used through the :meth:`register_card` decorator.

    Examples
    --------
    >>> @CardManager.register_card
    ... class ExampleCard:
    ...     pass
    >>> CardManager.card_info_dict['ExampleCard'].__name__
    'ExampleCard'
    """

    card_info_dict = {}
    card_metadata_dict: dict[str, CardMetadata] = {}

    @classmethod
    def register_card(cls, card_class):
        """Register a card class, keyed by its class name.

        Parameters
        ----------
        card_class : type
            Class object to be registered.

        Returns
        -------
        type
            The same class, to support decorator usage.
        """
        if card_class.__name__ in cls.card_info_dict:
            logger.warning(f"The registered Card class name {card_class.__name__} is duplicated. The most recently registered one will be used.")
        cls.card_info_dict[card_class.__name__] = card_class
        cls.card_metadata_dict[card_class.__name__] = build_card_metadata(card_class)
        return card_class

    @classmethod
    def get_card_metadata(cls, class_name: str) -> CardMetadata | None:
        """Return metadata for a registered card class."""
        return cls.card_metadata_dict.get(class_name)




def load_cards_from_directory(directory: str):
    """Load and import all card modules within a directory.

    Parameters
    ----------
    directory : str
        Folder path to scan for Python modules. Files starting with ``_`` are
        ignored.

    Returns
    -------
    None
        The function imports modules for their side effect of registration.

    Notes
    -----
    Each module is expected to register its cards using the
    :meth:`CardManager.register_card` decorator at import time.

    Examples
    --------
    >>> load_cards_from_directory('path/to/cards')  # doctest: +SKIP
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        return None

    for file_path in dir_path.glob("*.py"):

        if file_path.name.startswith("_"):
            continue  # Skip private/python module files

        module_name = file_path.stem
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # The module should register its cards automatically via decorators
            logger.success(f"Successfully loaded card module: {module_name}")


        except Exception as e:
            logger.error(f"Failed to load card module {file_path}: {str(e)}")

    return None
