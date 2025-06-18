import importlib
from pathlib import Path

from loguru import logger

from NepTrainKit.core import MessageManager


class CardManager:
    card_info_dict = {}

    @classmethod
    def register_card(cls,card_class):
        if card_class.__name__   in cls.card_info_dict:
            logger.warning(f"The registered Card class name {card_class.__name__} is duplicated. The most recently registered one will be used.")
        cls.card_info_dict[card_class.__name__] = card_class
        return card_class




def load_cards_from_directory(directory: str):
    """Load all card modules from a directory"""
    dir_path = Path(directory)

    if not dir_path.is_dir():
        return None
    #     raise ValueError(f"Directory not found: {directory}")

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