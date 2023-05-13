# Logger that takes periodically takes dictionaries and saves them to csv.
import json
import pathlib
from collections import defaultdict


class Logger:
    # Accumulate data dictionaries and save them to csv.
    def __init__(self, checkpoints_path: str, seed: int):
        root = pathlib.Path(checkpoints_path)
        root.mkdir(parents=True, exist_ok=True)
        # make 2 files for train and eval
        self._train_file = root / f"train_{seed}.json"
        self._eval_file = root / f"eval_{seed}.json"
        self._train_file.touch(exist_ok=True)
        self._eval_file.touch(exist_ok=True)
        self._train_log = defaultdict(list)
        self._eval_log = defaultdict(list)

    def log(self, data: dict, mode: str):
        assert mode in ["train", "eval"]
        file = self._train_file if mode == "train" else self._eval_file
        log = self._train_log if mode == "train" else self._eval_log

        # Accumulate data.
        for key, value in data.items():
            log[key].append(value)

        # Dump data to disk.
        with open(file, "w") as f:
            json.dump(log, f, indent=4)
