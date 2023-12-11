import os
import argparse

import torch
import tqdm
import pandas as pd
from omegaconf import OmegaConf
from sklearn import metrics
from transformers import AutoFeatureExtractor

from models.model_wrapper import PlModelWrapper
from utils.dataloader import Wav2vec2CutomDataset, CollateFuncWav2vec2
from utils.evaluate import test_model

from utils.utils import (
    preprocess_metadata,
    add_hierarchy_labels,
    save_conf_matrix
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

    label_column = "label"
    metadata_paths = [
            "../../new_data_5.0/Multidataset-ser/Metadata/metadata_test_audios_zap.csv",#7
            "../../new_data_5.0/Multidataset-ser/Metadata/metadata_test_coraa-ser.csv",#8
            "../../new_data_5.0/Multidataset-ser/Metadata/metadata_test_emoUERJ.csv",#9
            "../../new_data_5.0/Multidataset-ser/Metadata/metadata_test_ravdess.csv",#10
            "../../new_data_5.0/Multidataset-ser/Metadata/metadata_prs_labels_duplas_duplicadas.csv",#11
            "../../Datasets-demo/all_datasets-by-demos.csv" #12
        ]

    dataset_names = [
        "audios_zap_3.0", #7
        "coraa_ser_3.0", #8
        "emoUERJ_3.0", #9
        "ravdess_test_3.0", #10
        "metadata_prs_labels_duplas_duplicadas", #11
        "application_data"
    ]

    base_dirs = [
        "../../new_data_5.0",  #7
        "../../new_data_5.0", #8
        "../../new_data_5.0",  #9
        "../../new_data_5.0", #10
        "../../new_data_5.0", #11
        "../../Datasets-demo"
    ]

    for metadata_path, base_dir, dataset_name in tqdm.tqdm(zip(metadata_paths, base_dirs, dataset_names)):

        print(f"Testing {dataset_name} ... \n")
        df = pd.read_csv(metadata_path)

        label2id = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
        }

        id2label = {
            0: "neutral",
            1: "happy",
            2: "sad",
            3: "angry",
            4: "fear",
            5: "disgust",
            6: "surprise",
        }

        hierarquical_label2id = {
            "neutral": 0,
            "positive": 1,
            "negative": 2
        }

        df = add_hierarchy_labels(df, config.metadata.label_column)
        df = df[df[label_column]!="frustrated"]
        df = df[df[label_column]!="excited"]

        if dataset_name == "metadata_prs_labels_duplas_duplicadas":
            df = df[df["dataset"] == "reality - casamento as cegas"]

        test_dataset = preprocess_metadata(base_dir=config.data.base_dir, cfg=config, df=df)

        processor = AutoFeatureExtractor.from_pretrained(config.model.pretrained_model_path)

        test_dataset = Wav2vec2CutomDataset(
            dataset=test_dataset,
            sampling_rate=config.data.target_sampling_rate,
            max_audio_len=config.data.max_audio_len,
            apply_augmentation=False,
            audio_augmentator=None,
            apply_dbfs_norm=config.data.apply_dbfs_norm,
            target_dbfs=config.data.target_dbfs,
            rand_sampling=False,
            label2id=label2id,
            hierarquical_label2id=hierarquical_label2id,
            filepath_column=config.metadata.audio_path_column,
            label_column=config.metadata.label_column
        )

        data_collator = CollateFuncWav2vec2(processor)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.training.num_workers,
            collate_fn=data_collator
        )

        labels, pred_list = test_model(
            test_dataloader=test_loader,
            config=config,
            checkpoint_path="../checkpoints/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition-multidaset-5.0-10_epochs-Gain_Norm_True-data_augmentation_False-neutro_cut-no_pad/last-v2.ckpt",
            label_type="sentiment"
        )

        classes = list(set(labels + pred_list))

        classes = [id2label[class_n] for class_n in classes]

        save_conf_matrix(
            targets=labels,
            preds=pred_list,
            classes=classes,
            output_path=f"../../conf_m/hierarquical_sentiment_wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition-multidaset-5.0-10_epochs-Gain_Norm_True-data_augmentation_False-neutro_cut-no_pad-{dataset_name}.png"
        )

        # acc = metrics.accuracy_score(y_true=labels, y_pred=pred_list)
        # f1 = metrics.f1_score(y_true=labels, y_pred=pred_list, average='macro')
        # precision = metrics.precision_score(y_true=labels, y_pred=pred_list, average='macro')
        # recall = metrics.recall_score(y_true=labels, y_pred=pred_list, average='macro')

        # with open(os.path.join("../../scores", 'scores_hierarquical_sentiment_wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition-multidaset-5.0-10_epochs-Gain_Norm_True-data_augmentation_False-neutro_cut-no_pad.txt'), 'a+') as file:
        #         file.write(f'{dataset_name} | '\
        #                         f'Accuracy: {acc} | '\
        #                         f'Precision: {precision} | '\
        #                         f'Recall: {recall} | '\
        #                         f'F1 Score: {f1}\n')

       
if __name__ == "__main__":
    main()