from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from models.model_wrapper import PlModelWrapper

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(
        test_dataloader: Any,
        config: dict,
        checkpoint_path: str,
        label_type: str = "sentiment"
    ):
    """
    Predicts new data.

    ----
    Args:
        test_data: Path to csv file containing the paths to the audios files for prediction and its labels.

        batch_size: Mini-Batch size.

        checkpoint_path: Path to the file that contains the saved weight of the model trained.

        num_workers: Number of workers to use as paralel processing.

        use_amp: True to use Mixed precision and False otherwise.
    """
    # print(config)
    model = PlModelWrapper().load_from_checkpoint(checkpoint_path)
    model.to(device)

    model.freeze()

    pred_list = []
    labels = []

    print(checkpoint_path)

    with torch.no_grad():
        model.eval()

        for x, y_sentiment, y_language in tqdm(test_dataloader):
            test_audio, test_sentiment_label, hierarquical_label = x.to(device), y_sentiment.to(device), y_language.to(device)

            sentiment_logits, hierarchy_logits = model(test_audio)

            if label_type == "sentiment":
                pred = torch.argmax(sentiment_logits, axis=1).cpu().detach().numpy()
                label = test_sentiment_label.cpu().detach().numpy()

                pred_list.extend(pred)
                labels.extend(label)

            else:
                pred = torch.argmax(hierarchy_logits, axis=1).cpu().detach().numpy()
                label = hierarquical_label.cpu().detach().numpy()

                pred_list.extend(pred)
                labels.extend(label)

    return labels, pred_list