import os
import operator
import functools
import traceback
from typing import Dict, List, Optional, Union

import torch
import tqdm
import torch.nn.functional as F
import torchaudio
from sklearn import metrics

import numpy as np
import pandas as pd
import seaborn as sns
from pydub import AudioSegment
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from transformers import Trainer
from datasets import Dataset, load_metric
from transformers import Wav2Vec2Processor
from sklearn.metrics import confusion_matrix
from audiomentations import AddGaussianNoise, PitchShift, Mp3Compression, TimeMask, AddGaussianSNR
from torch.utils.data import WeightedRandomSampler, DataLoader

metric = load_metric("f1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_top2_acc(preds, labels):
    """
    Computes the top2 accuracy
    """
    top2_preds = torch.argsort(preds, dim=1, descending=True)[:, :2]
    labels = labels.unsqueeze(1).expand(top2_preds.size())
    correct = top2_preds.eq(labels)
    acc = correct.sum().float() / labels.size(0)
    return acc


def get_top2_pred(preds, labels):
    """
    Computes the top2 macro f1 score using sklearn
    """
    new_pred = []
    top2_preds = torch.argsort(preds, dim=1, descending=True)[:, :2]
    labels_temp = labels.unsqueeze(1).expand(top2_preds.size())
    correct = top2_preds.eq(labels_temp)

    result = torch.where(correct, top2_preds, -1)

    for i, j in zip(result, top2_preds):
        if i.sum() == -2:
            new_pred.append(j[0])
        else:
            new_pred.append(i[i != -1][0])

    return new_pred


def plot_pred_distribution(predictions: np.ndarray, save_path: str):
    """
    Plots and saves in png the distribution of the confidence of a model considering intervals of 10 buckets: 0.1, 0.2, 0.3 ... 1.0
    """

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get the confidence of the predictions
    confidences = np.max(predictions, axis=1)

    # Get the predicted labels
    predictions = np.argmax(predictions, axis=1)

    # Get the bin edges
    bin_edges = np.linspace(0., 1. + 1e-8, 11)

    # Get the bin indices
    bin_indices = np.digitize(confidences, bin_edges)

    # Get the number of predictions in each bin
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(1, len(bin_edges))])
    
    # Plot the distribution
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], bin_counts, width=0.1, align='edge', color='blue', alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Number of predictions')
    plt.title('Distribution of the confidence of the predictions')
    plt.savefig(save_path)
    plt.close()


def plot_pred_distribution_per_label(predictions: np.ndarray, save_path: str):
    """
    Plots and saves in png the distribution of the confidence of a model per label considering intervals of 10 buckets: 0.1, 0.2, 0.3 ... 1.0 com 
    as labels  neutral, happy, sad, angry, fear, disgust, surprise. Each label must be in a different image.
    """

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get the confidence of the predictions
    confidences = np.max(predictions, axis=1)

    # Get the predicted labels
    predictions = np.argmax(predictions, axis=1)

    # Get the bin edges
    bin_edges = np.linspace(0., 1. + 1e-8, 11)

    # Get the bin indices
    bin_indices = np.digitize(confidences, bin_edges)

    # Get the number of predictions in each bin
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(1, len(bin_edges))])

   # Get the labels
    labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

    # Get the indices of the predictions for each label
    indices = [np.where(predictions == i)[0] for i in range(len(labels))]

    # Get the confidences of the predictions for each label
    confidences_per_label = [confidences[indices[i]] for i in range(len(labels))]

    # Get the bin indices for each label
    bin_indices_per_label = [np.digitize(confidences_per_label[i], bin_edges) for i in range(len(labels))]

    # Get the number of predictions in each bin for each label
    bin_counts_per_label = [np.array([np.sum(bin_indices_per_label[i] == j) for j in range(1, len(bin_edges))]) for i in range(len(labels))]

    # Plot the distribution for each label
    for i in range(len(labels)):
        plt.figure(figsize=(10, 5))
        plt.bar(bin_edges[:-1], bin_counts_per_label[i], width=0.1, align='edge', color='blue', alpha=0.5)
        plt.xlabel('Confidence')
        plt.ylabel('Number of predictions')
        plt.title(f'Distribution of the confidence of the predictions for the label {labels[i]}')
        plt.savefig(save_path.replace(".png", f"_{labels[i]}.png"))
        plt.close()


def convert_labels(df, label_column):
    emotion2int = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    df = df[df[label_column]!="frustrated"]
    df = df[df[label_column]!="excited"]

    df = df.replace({label_column: emotion2int})

    return df


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
    

def compute_metrics(eval_pred):
    """Computes metric on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")


def map_path(batch, base_dir, cfg):
    """Maps the real path to the audio files"""
    path = os.path.join(base_dir, batch[cfg.metadata.audio_path_column].lstrip('/'))
    batch['input_values'] = path
    return batch


def preprocess_metadata(base_dir: str, cfg: DictConfig, df: pd.DataFrame):
    """Maps the real path to the audio files"""
    df.reset_index(drop=True, inplace=True)

    df_dataset = Dataset.from_pandas(df)
    df_dataset = df_dataset.map(
        map_path,
        fn_kwargs={"base_dir": base_dir, "cfg": cfg},
        num_proc=cfg.train.num_workers
    )

    print(df_dataset)

    return df_dataset

def reliability_diagram(y_true, y_pred, n_bins=10, title='Reliability Diagram', save_path=None):
    """
    Plot a reliability diagram
    :param y_true: true labels
    :param y_pred: predicted labels
    :param n_bins: number of bins
    :param title: plot title
    :param save_path: path to save the plot
    :return:
    """

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get the confidence of the predictions
    confidences = np.max(y_pred, axis=1)

    # Get the predicted labels
    y_pred = np.argmax(y_pred, axis=1)

    # print(y_pred, type(y_pred))

    # # Get the true labels
    # y_true = np.argmax(y_true, axis=1)

    y_true = np.array(y_true)

    # Get the bin edges
    bin_edges = np.linspace(0., 1. + 1e-8, n_bins + 1)

    # Get the bin centers
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)])

    # Get the bin indices for each prediction
    bin_indices = np.digitize(confidences, bin_edges[:-1])

    # Get the number of predictions in each bin
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(1, len(bin_edges))])

    # Get the average confidence in each bin
    bin_confidences = np.array([np.mean(confidences[bin_indices == i]) for i in range(1, len(bin_edges))])

    # Get the accuracy in each bin
    bin_accuracies = np.array([np.mean(y_pred[bin_indices == i] == y_true[bin_indices == i]) for i in range(1, len(bin_edges))])

    # Plot the reliability diagram
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.plot(bin_confidences, bin_accuracies, 's-', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def get_label_id(dataset: Dataset, label_column: str):
    """Gets the labels IDs"""
    label2id, id2label = dict(), dict()

    labels = dataset.unique(label_column)
    labels.sort()

    num_labels = len(id2label)

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return label2id, id2label, num_labels


def predict(test_dataloader, model, cfg):
    model.to(device)
    model.eval()
    preds = []
    paths = []
    logits_list = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            input_values, attention_mask = batch['input_values'].to(device), batch['attention_mask'].to(device)

            logits = model(input_values, attention_mask=attention_mask).logits
            scores = F.softmax(logits, dim=-1)

            logits_list.extend(scores.cpu().detach().numpy())

            pred = torch.argmax(scores, dim=1).cpu().detach().numpy()

            preds.extend(pred)

    return preds, logits_list


def map_data_augmentation(aug_config):
    # Adapted from: https://github.com/Edresson/Wav2Vec-Wrapper
    aug_name = aug_config.name
    del aug_config.name
    if aug_name == 'gaussian':
        return AddGaussianNoise(**aug_config)
    elif aug_name == 'gaussian_snr':
        return AddGaussianSNR(**aug_config)
    elif aug_name == 'mp3_compression':
        return Mp3Compression(**aug_config)
    elif aug_name == 'time_mask':
        return TimeMask(**aug_config)

    else:
        raise ValueError("The data augmentation '" + aug_name + "' doesn't exist !!")

class DataColletorTrain:
    # Adapted from https://github.com/Edresson/Wav2Vec-Wrapper
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        apply_augmentation: bool = False,
        audio_augmentator: List[Dict] =  None,
        sampling_rate: int = 16000,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        apply_dbfs_norm: Union[bool, str] = False,
        target_dbfs: int = 0.0,
        label2id: Dict = None,
        max_audio_len: int = 20
    ):

        self.processor = processor
        self.sampling_rate = sampling_rate

        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.apply_dbfs_norm = apply_dbfs_norm
        self.target_dbfs = target_dbfs

        self.apply_augmentation = apply_augmentation
        self.audio_augmentator = audio_augmentator

        self.label2id = label2id

        self.max_audio_len = max_audio_len

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = []
        label_features = []
        for feature in features:
            try:
                # Gain Normalization
                if self.apply_dbfs_norm:
                    # Audio is loaded in a byte array
                    sound = AudioSegment.from_file(feature["input_values"], format="wav")
                    sound = sound.set_channels(1)
                    change_in_dBFS = self.target_dbfs - sound.dBFS
                    # Apply normalization
                    normalized_sound = sound.apply_gain(change_in_dBFS)
                    # Convert array of bytes back to array of samples in the range [-1, 1]
                    # This enables to work wih the audio without saving on disk
                    norm_audio_samples = np.array(normalized_sound.get_array_of_samples()).astype(np.float32, order='C') / 32768.0

                    if sound.channels < 2:
                        norm_audio_samples = np.expand_dims(norm_audio_samples, axis=0)

                    # Expand one dimension and convert to torch tensor to have the save output shape and type as torchaudio.load
                    speech_array = torch.from_numpy(norm_audio_samples)
                    sampling_rate = sound.frame_rate

                # Load wav
                else:
                    speech_array, sampling_rate = torchaudio.load(feature["input_values"])

                # Transform to Mono
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)

                if sampling_rate != self.sampling_rate:
                    transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                    speech_array = transform(speech_array)
                    sampling_rate = self.sampling_rate

                effective_size_len = sampling_rate * self.max_audio_len

                if speech_array.shape[-1] > effective_size_len:
                    speech_array = speech_array[:, :effective_size_len]

                speech_array = speech_array.squeeze().numpy()
                input_tensor = self.processor(speech_array, sampling_rate=sampling_rate).input_values
                input_tensor = np.squeeze(input_tensor)

                if self.audio_augmentator is not None and self.apply_augmentation:
                    input_tensor = self.audio_augmentator(input_tensor, sample_rate=self.sampling_rate).tolist()

                input_features.append({"input_values": input_tensor})
                label_features.append(int(self.label2id[feature["label"]]))
            except Exception:
                print("Error during load of audio:", feature["input_values"])
                continue

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features)

        return batch


class DataColletorTest:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        sampling_rate: int = 16000,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label2id: Dict = None,
        max_audio_len: int = 20
    ):

        self.processor = processor
        self.sampling_rate = sampling_rate

        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.label2id = label2id

        self.max_audio_len = max_audio_len

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = []
        label_features = []
        for feature in features:
            # Load wav
            speech_array, sampling_rate = torchaudio.load(feature["input_values"])

            # Transform to Mono
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

            if sampling_rate != self.sampling_rate:
                transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                speech_array = transform(speech_array)
                sampling_rate = self.sampling_rate

            effective_size_len = sampling_rate * self.max_audio_len

            if speech_array.shape[-1] > effective_size_len:
                speech_array = speech_array[:, :effective_size_len]

            speech_array = speech_array.squeeze().numpy()
            input_tensor = self.processor(speech_array, sampling_rate=sampling_rate).input_values
            input_tensor = np.squeeze(input_tensor)

            input_features.append({"input_values": input_tensor})
            label_features.append(int(self.label2id[feature["label"]]))

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features)

        return batch


class Wav2vec2CutomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: dict,
        sampling_rate: int = 16000,
        max_audio_len: int = 20,
        apply_augmentation: bool = False,
        audio_augmentator: List[Dict] =  None,
        apply_dbfs_norm: Union[bool, str] = False,
        target_dbfs: int = 0.0,
        rand_sampling: bool = False,
        label2id: Dict = None,
        filepath_column: str = "filepath",
        label_column: str = "label"
    ):
        self.dataset = dataset

        self.sampling_rate = sampling_rate

        self.apply_dbfs_norm = apply_dbfs_norm
        self.target_dbfs = target_dbfs
        self.apply_augmentation = apply_augmentation
        self.rand_sampling = rand_sampling
        self.audio_augmentator = audio_augmentator

        self.label2id = label2id

        self.max_audio_len = max_audio_len

        self.filepath_column = filepath_column
        self.label_column = label_column

    def __len__(self):
        return self.dataset.num_rows

    def __cutorpad(self, audio: np.ndarray) -> np.ndarray:
        """
        Cut or pad an audio
        """
        effective_length = self.sampling_rate * self.max_audio_len
        len_audio = len(audio)

        if self.rand_sampling:
            # If audio length is less than wished audio length
            # if len_audio < effective_length:
            #     new_audio = np.zeros(effective_length)
            #     start = np.random.randint(effective_length - len_audio)
            #     new_audio[start:start + len_audio] = audio
            #     audio = new_audio

            # If audio length is bigger than wished audio length
            if len_audio > effective_length:
                start = np.random.randint(len_audio - effective_length)
                audio = audio[start:start + effective_length]

            # If audio length is equal to wished audio length
            # else:
            #     audio = audio

        else :
            # # If audio length is less than wished audio length
            # if len_audio < effective_length:
            #     new_audio = np.zeros(effective_length)
            #     new_audio[:len_audio] = audio
            #     audio = new_audio

            # If audio length is bigger than wished audio length
            if len_audio > effective_length:
                audio = audio[:effective_length]

            # If audio length is equal to wished audio length
            # else:
            #     audio = audio

        # Expand one dimension related to the channel dimension
        return audio


    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        # Gain Normalization
        filepath = self.dataset[index]["input_values"]
        label = int(self.label2id[self.dataset[index][self.label_column]])

        if self.apply_dbfs_norm:
            # Audio is loaded in a byte array
            sound = AudioSegment.from_file(filepath, format="wav")
            sound = sound.set_channels(1)
            change_in_dBFS = self.target_dbfs - sound.dBFS
            # Apply normalization
            normalized_sound = sound.apply_gain(change_in_dBFS)
            # Convert array of bytes back to array of samples in the range [-1, 1]
            # This enables to work wih the audio without saving on disk
            norm_audio_samples = np.array(normalized_sound.get_array_of_samples()).astype(np.float32, order='C') / 32768.0

            if sound.channels < 2:
                norm_audio_samples = np.expand_dims(norm_audio_samples, axis=0)

            # Expand one dimension and convert to torch tensor to have the save output shape and type as torchaudio.load
            speech_array = torch.from_numpy(norm_audio_samples)
            sr = sound.frame_rate

        # Load wav
        else:
            speech_array, sr = torchaudio.load(filepath)

        # Transform to Mono
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech_array = transform(speech_array)
            sr = self.sampling_rate

        speech_array = speech_array.squeeze().numpy()

        # Cut or pad audio
        speech_array = self.__cutorpad(speech_array)

        if self.audio_augmentator is not None and self.apply_augmentation:
            # print("> Applying Data Augmentation...")
            speech_array = self.audio_augmentator(speech_array, sample_rate=self.sampling_rate).tolist()
        
        return speech_array, label

class CollateFuncWav2vec2:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        sampling_rate: int = 16000,
    ):
        self.padding = padding
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List):
        label_features = []
        input_features = []

        for audio, label in batch:
            input_tensor = self.processor(audio, sampling_rate=self.sampling_rate).input_values
            input_tensor = np.squeeze(input_tensor)
            input_features.append({"input_values": input_tensor})
            label_features.append(int(label))

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features)

        return batch


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


def gen_sample_weights(dataset: dict, label2id: str, label_column: str ='label'):
    """
    Generate sample weights for a dataset
    """
    # Get the count of each label
    label_count = np.zeros(len(label2id))
    for audio, label in tqdm.tqdm(dataset):
        # label = int(label2id[element[label_column]])
        label_count[label] += 1

    # The reason for 1000.0 is to make sure that the weights are not too small and 0.01 is to avoid division by 0
    label_weight = 1000.0 / (label_count + 0.01)
    sample_weight = np.zeros(len(dataset))

    print(label_count)
    print(label_weight)

    # Assign weights to each sample
    for element in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        idx, (audio, label) = element
        # label = int(label2id[element[label_column]])
        sample_weight[idx] = label_weight[label]

    return sample_weight

def gen_sample_weights_2(dataset: dict, label2id: str, label_column: str ='label'):
    """
    Generate sample weights for a dataset
    """
    # Get the count of each label
    label_count = np.zeros(len(label2id))
    for element in tqdm.tqdm(dataset):
        label = int(label2id[element[label_column]])
        label_count[label] += 1

    # The reason for 1000.0 is to make sure that the weights are not too small and 0.01 is to avoid division by 0
    label_weight = 1000.0 / (label_count + 0.01)
    sample_weight = np.zeros(len(dataset))

    print(label_count)
    print(label_weight)

    # Assign weights to each sample
    for element in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        idx, element_ins = element
        label = int(label2id[element_ins[label_column]])
        # label = int(label2id[element[label_column]])
        sample_weight[idx] = label_weight[label]

    return sample_weight


class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        label2id = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "fear": 4,
            "disgust": 5,
            "surprise": 6,
        }

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        samples_weight = gen_sample_weights(train_dataset, label2id=label2id)
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )

