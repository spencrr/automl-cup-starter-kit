from pathlib import Path
from dataloader import AutoMLCupDataloader
from datasets import load_dataset, DatasetDict


class SpliceDataloader(AutoMLCupDataloader):
    @staticmethod
    def name():
        return "splice"

    val_size = 0.1
    test_size = 0.1
    shuffle = False

    def __init__(self, directory: Path, **kwargs):
        super().__init__(directory, **kwargs)

        cache_directory = directory / "cache"
        if cache_directory.exists():
            self.dataset = DatasetDict.load_from_disk(cache_directory)
        else:

            def positions_to_vec(example):
                example["input"] = list(example[f"position_{i}"] for i in range(60))
                return example

            train_val_test_dataset = (
                load_dataset(
                    "mstz/splice",
                    "splice",
                    data_dir=directory / SpliceDataloader.name(),
                )["train"]
                .map(
                    positions_to_vec,
                    remove_columns=list(f"position_{i}" for i in range(60)),
                )
                .rename_column("class", "label")
                .train_test_split(
                    test_size=SpliceDataloader.val_size + SpliceDataloader.test_size,
                    shuffle=SpliceDataloader.shuffle,
                )
            )
            val_test_dataset = train_val_test_dataset["test"].train_test_split(
                test_size=(
                    SpliceDataloader.test_size
                    / (SpliceDataloader.val_size + SpliceDataloader.test_size)
                ),
                shuffle=SpliceDataloader.shuffle,
            )

            self.dataset = DatasetDict(
                {
                    "train": train_val_test_dataset["train"],
                    "val": val_test_dataset["train"],
                    "test": val_test_dataset["test"],
                }
            )

            self.dataset.save_to_disk(cache_directory)

    def get_split(self, split):
        if split in ["train", "val", "test"]:
            return self.dataset[split]
