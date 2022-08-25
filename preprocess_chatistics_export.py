import glob
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def add_speaker_tag(
    row, speaker1_tag="<speaker1>", speaker2_tag="<speaker2>", speaker_tag_sep=" "
):
    if row["should_add_name_tag"]:
        if row["outgoing"]:
            speaker_tag = speaker1_tag
        else:
            speaker_tag = speaker2_tag
        return speaker_tag_sep.join([speaker_tag, row["text"]])
    return row["text"]


def transform_to_text(df_path, min_messages: int = 10, max_messages: int = 100000):
    logger.info("loading data...")
    df = pd.read_json(df_path)
    logger.info("data loaded")

    df = df[df["language"] == "pl"]

    for group_name, group in df.groupby("conversationWithName"):
        if group.shape[0] < min_messages:
            logger.info(
                f"skipping conversation with:{group_name}, num of messages:{group.shape[0]} (<10 messages)"
            )
            continue
        if group["senderName"].nunique() != 2:
            logger.info(
                f"skipping conversation with:{group_name}, num of messages:{group.shape[0]} (group or solo conversation)"
            )
            continue
        logger.info(
            f"processing conversation with:{group_name}, num of messages:{group.shape[0]}"
        )
        if group.shape[0] > max_messages:
            logger.info(f"truncating group to {max_messages} messages")
            group = group[:max_messages]

        group = group.sort_values("timestamp")
        group["should_add_name_tag"] = group["senderName"].ne(
            group["senderName"].shift()
        )
        texts = group.apply(add_speaker_tag, axis=1)
        concatenated_texts = texts.str.cat(sep=" ")
        concatenated_texts_with_speaker_newlines = re.sub(
            r" (?=<speaker[12]>)", "\n", concatenated_texts
        )
        yield concatenated_texts_with_speaker_newlines


@click.command()
@click.option(
    "--chatistics_export_path",
    type=Path,
)
@click.option(
    "--output_path",
    type=Path,
    default=Path(f"./your_text_messages_{datetime.now().strftime('%Y_%M_%d_%H_%M')}.txt"),
)
def main(chatistics_export_path: Path, output_path: Path):
    if not chatistics_export_path:
        list_of_files = glob.glob("./Chatistics/exports/*")
        latest_file = max(list_of_files, key=os.path.getctime)
        chatistics_export_path = Path(latest_file)

    logger.info(f"chatistics export path: {chatistics_export_path}")
    texts_gen = transform_to_text(chatistics_export_path)
    with open(output_path, "w") as f:
        f.writelines(texts_gen)


if __name__ == "__main__":
    main()
