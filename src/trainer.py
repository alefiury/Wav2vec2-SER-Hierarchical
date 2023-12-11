import os
import argparse

import torch
import tqdm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from audiomentations import Compose
from transformers import AutoFeatureExtractor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from models.model_wrapper import PlModelWrapper
from utils.dataloader import Wav2vec2CutomDataset, CollateFuncWav2vec2
from utils.evaluate import test_model
from utils.utils import (
    undersample_majority_class,
    preprocess_metadata,
    map_data_augmentation,
    add_hierarchy_labels,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        default=os.path.join("../", "config", "default.yaml"),
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument("-g", "--gpu", required=True, type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    pl.seed_everything(42)

    pl_model = PlModelWrapper(**config.model)
    print(pl_model)

    wandb_logger = WandbLogger(
        name=os.path.basename(config.model_checkpoint.dirpath),
        project=config.logger.project_name,
        entity="alefiury",
        mode="disabled" if config.logger.debug else None
    )

    if config.data.audio_augmentator and config.data.apply_augmentation:
        print("> Applying Data Augmentation...")
        audio_augmentator = Compose([map_data_augmentation(aug_config) for aug_config in config.data.audio_augmentator])
    else:
        audio_augmentator = None

    train_df = pd.read_csv(config.metadata.train_metadata_path)

    if config.metadata.val_metadata_path is None:
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=train_df[[config.metadata.label_column]]
        )
    else:
        val_df = pd.read_csv(config.metadata.val_metadata_path)


    train_df = train_df[train_df[config.metadata.label_column]!="frustrated"]
    train_df = train_df[train_df[config.metadata.label_column]!="excited"]

    val_df = val_df[val_df[config.metadata.label_column]!="frustrated"]
    val_df = val_df[val_df[config.metadata.label_column]!="excited"]

    train_df = add_hierarchy_labels(train_df, config.metadata.label_column)
    val_df = add_hierarchy_labels(val_df, config.metadata.label_column)

    train_df = train_df[train_df[config.metadata.dataset_column]!="ANAD"]
    val_df = val_df[val_df[config.metadata.dataset_column]!="ANAD"]

    train_df = train_df[train_df[config.metadata.dataset_column]!="MOSEI"]
    val_df = val_df[val_df[config.metadata.dataset_column]!="MOSEI"]

    print("-"*50)
    print(train_df["dataset"].unique())
    print(val_df["dataset"].unique())
    print("-"*50)
    print(train_df["label"].unique())
    print(val_df["label"].unique())
    print("-"*50)
    print(train_df["language"].unique())
    print(val_df["language"].unique())
    print("-"*50)

    train_df = undersample_majority_class(
        train_df, 
        config.metadata.label_column, 
        config.metadata.language_column,
        "neutral"
    )

    print(train_df[config.metadata.label_column].value_counts().to_dict())

    train_dataset = preprocess_metadata(base_dir=config.data.base_dir, cfg=config, df=train_df)
    val_dataset = preprocess_metadata(base_dir=config.data.base_dir, cfg=config, df=val_df)

    label2id = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    hierarquical_label2id = {
        "neutral": 0,
        "positive": 1,
        "negative": 2
    }

    print("/"*200)

    print(train_dataset.num_rows/16)

    processor = AutoFeatureExtractor.from_pretrained(config.model.pretrained_model_path)

    train_dataset = Wav2vec2CutomDataset(
        dataset=train_dataset,
        sampling_rate=config.data.target_sampling_rate,
        max_audio_len=config.data.max_audio_len,
        apply_augmentation=config.data.apply_augmentation,
        audio_augmentator=audio_augmentator,
        apply_dbfs_norm=config.data.apply_dbfs_norm,
        target_dbfs=config.data.target_dbfs,
        rand_sampling=config.data.rand_sampling,
        label2id=label2id,
        hierarquical_label2id=hierarquical_label2id,
        filepath_column=config.metadata.audio_path_column,
        label_column=config.metadata.label_column
    )

    val_dataset = Wav2vec2CutomDataset(
        dataset=val_dataset,
        sampling_rate=config.data.target_sampling_rate,
        max_audio_len=config.data.max_audio_len,
        apply_augmentation=False,
        audio_augmentator=None,
        apply_dbfs_norm=False,
        target_dbfs=config.data.target_dbfs,
        rand_sampling=False,
        label2id=label2id,
        hierarquical_label2id=hierarquical_label2id,
        filepath_column=config.metadata.audio_path_column,
        label_column=config.metadata.label_column
    )

    data_collator = CollateFuncWav2vec2(processor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.training.num_workers,
        collate_fn=data_collator
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.training.num_workers,
        collate_fn=data_collator
    )

    trainer = pl.Trainer(
        **config.trainer,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(**config.model_checkpoint),
            LearningRateMonitor("step")
        ],
        devices=[args.gpu]
    )
    trainer.fit(pl_model, train_loader, val_loader)

if __name__ == "__main__":
    main()