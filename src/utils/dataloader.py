from typing import Any, Optional, List, Dict, Union

import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
import torch.nn.functional as F
from transformers import Wav2Vec2Processor


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
        hierarquical_label2id: Dict = None,
        filepath_column: str = "filepath",
        label_column: str = "label"
    ):
        """
        Wav2vec2CutomDataset
        """
        super().__init__()

        self.dataset = dataset

        self.sampling_rate = sampling_rate

        self.apply_dbfs_norm = apply_dbfs_norm
        self.target_dbfs = target_dbfs
        self.apply_augmentation = apply_augmentation
        self.rand_sampling = rand_sampling
        self.audio_augmentator = audio_augmentator

        self.label2id = label2id
        self.hierarquical_label2id = hierarquical_label2id

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
            # If audio length is bigger than wished audio length
            if len_audio > effective_length:
                start = np.random.randint(len_audio - effective_length)
                audio = audio[start:start + effective_length]
        else :
            # If audio length is bigger than wished audio length
            if len_audio > effective_length:
                audio = audio[:effective_length]

        # Expand one dimension related to the channel dimension
        return audio


    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        # Gain Normalization
        filepath = self.dataset[index]["input_values"]
        sentiment_label = int(self.label2id[self.dataset[index][self.label_column]])
        hierarquical_label = int(self.hierarquical_label2id[self.dataset[index]["hierarchy_label"]])

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

        return speech_array, sentiment_label, hierarquical_label

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
        sentiment_labels = []
        hierarquical_labels = []
        input_features = []

        for audio, sentiment_label, hierarquical_label in batch:
            input_tensor = self.processor(audio, sampling_rate=self.sampling_rate).input_values
            input_tensor = np.squeeze(input_tensor)
            input_features.append({"input_values": input_tensor})

            sentiment_labels.append(int(sentiment_label))
            hierarquical_labels.append(int(hierarquical_label))

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sentiment_labels = torch.tensor(sentiment_labels)
        hierarquical_labels = torch.tensor(hierarquical_labels)

        return batch, sentiment_labels, hierarquical_labels

