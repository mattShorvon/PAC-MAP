"""IO actions include importing and exporting a .spn file, with a representation
particular to this implementation"""

from typing import Mapping, Any
import logging
from pathlib import Path
from spn.io.file import to_file, from_file
from spn.io.zhao_file import to_zhao_file, from_zhao_file
from database import DB
from spn.actions.base import Action
from spn.data.parsed_data import ParsedData


class Export(Action):
    necessary_params = ["spn", "filename"]
    key = "export"

    def execute(self):
        """Exports the spn to a .spn file"""
        logging.info("Executing Export action")
        spn = DB.get(self.params["spn"])
        to_file(spn, Path(self.params["filename"]))


class Load(Action):
    necessary_params = ["filename", "name"]
    key = "load"

    def execute(self):
        """Imports the spn from a .spn file"""
        logging.info("Executing Load action")
        spn = from_file(Path(self.params["filename"]))
        # We add a None there because there is no Data
        DB.store(self.params["name"], spn)


class LoadData(Action):
    necessary_params = ["filename", "name"]
    key = "load-data"

    def execute(self):
        """Imports a dataset file"""
        logging.info("Executing Load-Data action")
        data = ParsedData(Path(self.params["filename"]))
        DB.store(self.params["name"], data)


class SaveData(Action):
    necessary_params = ["filename", "name"]
    key = "save-data"

    def execute(self):
        """Exports a dataset file"""
        logging.info("Executing Save-Data action")
        data = DB.get(self.params["name"])
        with open(self.params["filename"], "w") as f:
            for var in data.scope:
                f.write(f"var {var.id} {var.n_categories}\n")
            for data_line in data.data:
                line = ""
                for number in data_line:
                    line += f"{number} "
                line = line.strip()
                f.write(f"{line}\n")


class MultiLoad(Action):
    necessary_params = ["folder"]
    key = "load-folder"

    def execute(self):
        """Imports all .spn files from a folder"""
        logging.info("Executing Load Folder action")
        for filename in Path(self.params["folder"]).glob("*.spn"):
            spn = from_file(filename)
            # We add a None there because there is no Data
            spn_name = str(filename).split(self.params["folder"])[1].split(".spn")[0]
            DB.store(spn_name, (spn, None))


class ZhaoSave(Action):
    necessary_params = ["spn", "filename"]
    key = "export-zhao"

    def execute(self):
        """Exports an SPN to a file on Zhao's format for:
        https://github.com/KeiraZhao/SPN"""
        logging.info("Executing Export Zhao action")
        spn = DB.get(self.params["spn"])
        to_zhao_file(spn, Path(self.params["filename"]))


class ZhaoLoad(Action):
    necessary_params = ["filename", "name"]
    key = "load-zhao"

    def execute(self):
        """Imports an SPN from a file on Zhao's format for:
        https://github.com/KeiraZhao/SPN"""
        logging.info("Executing Load Zhao action")
        spn = from_zhao_file(self.params["filename"])
        DB.store(self.params["name"], spn)


class ExportDataset(Action):
    necessary_params = ["dataset", "filename", "pyspn-format"]
    key = "export-dataset"

    def execute(self):
        """Exports a dataset to a file of maybe different formats:"""
        logging.info("Executing Export dataset action")
        dataset = DB.get(self.params["dataset"])
        filename_path = self.params["filename"]
        dataset.to_file(filename_path, self.params["pyspn-format"])
