import os
from typing import List, List

import pandas as pd
import seaborn as sns
from datasets import Dataset
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix
# from audiomentations import AddGaussianNoise, Mp3Compression, TimeMask, AddGaussianSNR

# def map_data_augmentation(aug_config):
#     # Adapted from: https://github.com/Edresson/Wav2Vec-Wrapper
#     aug_name = aug_config.name
#     del aug_config.name
#     if aug_name == 'gaussian':
#         return AddGaussianNoise(**aug_config)
#     elif aug_name == 'gaussian_snr':
#         return AddGaussianSNR(**aug_config)
#     elif aug_name == 'mp3_compression':
#         return Mp3Compression(**aug_config)
#     elif aug_name == 'time_mask':
#         return TimeMask(**aug_config)

#     else:
#         raise ValueError("The data augmentation '" + aug_name + "' doesn't exist !!")


def undersample_majority_class(df, label_column, language_column, class_name):
    """Given a dataset, undersample the majority class considering the class with the second biggest number of samples"""

    # Get the number of samples of each class
    n_samples_classes = df[label_column].value_counts().to_dict()

    # Get the number of samples of the second biggest language that has the class_name
    n_samples_class_language = df[df[label_column]==class_name][language_column].value_counts().to_dict()

    # Get the language with the biggest number of samples for the class_name
    majority_class_language = list(n_samples_class_language.keys())[0]

    n_samples_second_majority_class_language = n_samples_class_language[list(n_samples_class_language.keys())[1]]

    df_majority_class = df[(df[label_column]==class_name) & (df[language_column]==majority_class_language)].sample(n=n_samples_second_majority_class_language, random_state=42)

    df_majority_class_n = df[(df[label_column]!=class_name) & (df[language_column]==majority_class_language)]

    # concat df_majority_class and df_majority_class_n
    df_majority_language = pd.concat([df_majority_class, df_majority_class_n])

    df_majority_language_n = df[(df[language_column]!=majority_class_language)]

    df_majority = pd.concat([df_majority_language, df_majority_language_n])


    return df_majority


def map_path(batch, base_dir, cfg):
    """Maps the real path to the audio files"""
    path = os.path.join(base_dir, batch[cfg.metadata.audio_path_column].lstrip('/'))
    batch['input_values'] = path
    return batch


def add_hierarchy_labels(df, label_column):
    hierarchy_dict = {
        "neutral": "neutral",
        "happy": "positive",
        "sad": "negative",
        "angry": "negative",
        "fear": "negative",
        "disgust": "negative",
        "surprise": "positive"
    }

    df["hierarchy_label"] = df[label_column]

    df["hierarchy_label"] = df["hierarchy_label"].map(hierarchy_dict)

    return df


def preprocess_metadata(base_dir: str, cfg: DictConfig, df: pd.DataFrame):
    """Maps the real path to the audio files"""
    df.reset_index(drop=True, inplace=True)

    df_dataset = Dataset.from_pandas(df)
    df_dataset = df_dataset.map(
        map_path,
        fn_kwargs={"base_dir": base_dir, "cfg": cfg},
        num_proc=cfg.training.num_workers
    )

    print(df_dataset)

    return df_dataset


def save_conf_matrix(
    targets: List[int],
    preds: List[int],
    classes: List[str],
    output_path: str
) -> None:
    """
    Saves a confusion matrix given the true labels and the predicted outputs.
    """
    cm = confusion_matrix(
        y_true=targets,
        y_pred=preds
    )

    df_cm = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    plt.figure(figsize=(24,12))
    plot = sns.heatmap(df_cm, annot=True,  fmt='g')
    figure1 = plot.get_figure()
    plot.set_ylabel('True Label')
    plot.set_xlabel('Predicted Label')
    plt.tight_layout()
    figure1.savefig(output_path, format='png')


def load_preloaded_data(config):
    if config.train_preloaded_path is None:
        return None, None, None

    preloaded_train_dataset = load_from_disk(config.train_preloaded_path)
    preloaded_val_dataset = load_from_disk(config.val_preloaded_path)
    preloaded_test_dataset = load_from_disk(config.test_preloaded_path)

    preloaded_train_dataset.set_format(
        type='torch',
        columns=[config.embedding_column, config.label_column]
    )
    preloaded_val_dataset.set_format(
        type='torch',
        columns=[config.embedding_column, config.label_column]
    )
    preloaded_test_dataset.set_format(
        type='torch',
        columns=[config.embedding_column, config.label_column]
    )


    return preloaded_train_dataset, preloaded_val_dataset, preloaded_test_dataset

def convert_labels(df, sentiment_label_column, language_label_column):
    # print(df[label_column].unique())
    emotion2int = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    language2int = {
        "english": 0,
        "urdu": 1,
        "italian": 2,
        "german": 3,
        "greek": 4
    }

    df = df[df[sentiment_label_column]!="frustrated"]
    df = df[df[sentiment_label_column]!="excited"]

    df = df.replace({sentiment_label_column: emotion2int})
    df = df.replace({language_label_column: language2int})

    return df

def convert_labels_iemocap(df, label_column):
    emotion2int = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    print(df[label_column].value_counts())
    df = df[df["label"].isin(["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"])]
    print(df[label_column].value_counts())
    df = df.replace({label_column: emotion2int})
    print(df[label_column].value_counts())

    return df


def convert_labels_coraa_ser(df, label_column):
    emotion2int = {
        "neutral": 0,
        "happiness": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    multiple_labels = {
        "happiness/anger": "happiness",
        "*neutral": "neutral",
        "happiness/surprise": "happiness",
        "sadness/happiness": "sadness",
        "happiness/fear": "happiness",
        "surprise/happiness": "surprise",
        "happiness/sadness": "happiness",
        "*anger": "anger"
    }

    df = df.replace({label_column: multiple_labels})
    print(df[label_column].value_counts())
    df = df.replace({label_column: emotion2int})
    print(df[label_column].value_counts())

    return df


def convert_metadata_to_preloaded(df, file_path_column, sufix, base_dir):
    df[file_path_column] = df[file_path_column].str.replace(base_dir, base_dir+f"_{sufix}")
    df[file_path_column] = df[file_path_column].str.replace(".wav", ".pt")

    return df

def remove_dataset(df, dataset_column, dataset_name):
    df_temp = df[df[dataset_column]!=dataset_name]

    return df_temp
