import os.path
import shutil
from .logger import logger

class DataStorage:
    def __init__(self, name:str, storage_root:str):
        self.name = name
        self.storage_root = storage_root

    def store(self, source:str):
        source_filename = source.split('/')[-1]
        filename, date_ext = source_filename.split('__')
        date, ext = date_ext.split('.')
        output_path = os.path.join(self.storage_root, self.name, date, ext.upper())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filename = f'{filename}.{ext}'
        output_path = os.path.join(output_path, filename)
        shutil.copy(source, output_path)
        logger.info(f"Raw file copied from Path: {source} to Path: {output_path}")