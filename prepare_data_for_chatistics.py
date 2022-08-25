import logging
import os
import re
import zipfile
from distutils.dir_util import copy_tree
from pathlib import Path

import click
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def unzip_facebook_messages(data_path: Path, output_path: Path):
    for filepath in tqdm(data_path.glob("*.zip"), desc="Extracting zip files"):
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            for el in zip_ref.namelist():
                if re.search(r"message_\d+.json", el):
                    zip_ref.extract(el, output_path)


def move_messages_to_chatistics(extracted_path: Path, chatistics_path: Path):
    dest_path = chatistics_path / "raw_data/messenger"
    extracted_messages_path = os.path.join(extracted_path, "messages")

    for msg_source_dir in tqdm(
        os.listdir(extracted_messages_path), desc="Copying files to Chatistics"
    ):
        copy_tree(
            os.path.join(extracted_messages_path, msg_source_dir), dest_path.as_posix()
        )


@click.command()
@click.option("--data_path", type=Path, default=Path("./data"))
@click.option("--extracted_path", type=Path, default=Path("./extracted"))
@click.option("--chatistics_path", type=Path, default=Path("./Chatistics"))
def main(data_path: Path, extracted_path: Path, chatistics_path: Path):
    logger.info(
        f"Extracting zip files, data_path: {data_path}, extracted_path: {extracted_path}"
    )

    unzip_facebook_messages(data_path, extracted_path)

    move_messages_to_chatistics(extracted_path, chatistics_path)
    # python parse.py messenger --max-exported-messages=1000000000
    # python export.py -f json


if __name__ == "__main__":
    main()
