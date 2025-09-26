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
import importlib
import importlib.util
from pathlib import Path
from loguru import logger

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
        return card_class




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
