import torch
from torch import nn
import pytorch_lightning as pl

from torchmetrics import Accuracy, MetricCollection
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)

from models.wav2vec2_model import Wav2Vec2ForSpeechClassification
from utils.scheduler import CosineWarmupLR

class PlModelWrapper(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_path: str = "facebook/wav2vec2-xls-r-300m",
        lr: float = 1e-4,
        alpha_loss: float = 0.5,
        warmup_ratio: float = 0.1,
        num_labels: int = 7
    ):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
            pretrained_model_path,
            num_labels=num_labels
        )

        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(
            pretrained_model_path,
            config=config,
        )

        self.lr = lr
        self.alpha_loss = alpha_loss
        self.warmup_ratio = warmup_ratio

        self.criterion = nn.CrossEntropyLoss()

        sentiment_metric_collection = MetricCollection([
            Accuracy(
                task="multiclass", 
                num_classes=num_labels
            ),
            MulticlassPrecision(
                num_classes=num_labels
            ),
            MulticlassRecall(
                num_classes=num_labels
            ),
            MulticlassF1Score(
                num_classes=num_labels
            )
        ])

        hierarchy_metric_collection = MetricCollection([
            Accuracy(
                task="multiclass", 
                num_classes=3
            ),
            MulticlassPrecision(
                num_classes=3
            ),
            MulticlassRecall(
                num_classes=3
            ),
            MulticlassF1Score(
                num_classes=3
            )
        ])

        self.train_sentiment_metrics = sentiment_metric_collection.clone(prefix='train-sentiment_')
        self.train_hierarchy_metrics = hierarchy_metric_collection.clone(prefix='train-hierarchy_')

        self.valid_sentiment_metrics = sentiment_metric_collection.clone(prefix='val-sentiment_')
        self.valid_hierarchy_metrics = hierarchy_metric_collection.clone(prefix='val-hierarchy_')

        self.test_sentiment_metrics = sentiment_metric_collection.clone(prefix='test-sentiment_')
        self.test_hierarchy_metrics = hierarchy_metric_collection.clone(prefix='test-hierarchy_')

    def get_hierarchy_logits(self, sentiment_logits):
        neutral_logits = sentiment_logits[:, 0]
        positive_logits = (sentiment_logits[:, 1] + sentiment_logits[:, 6]) / 2
        negative_logits = sentiment_logits[:, 2:6].float().mean(-1)

        hierarchy_logits = torch.stack([neutral_logits, positive_logits, negative_logits], dim=1)

        return hierarchy_logits


    def training_step(self, train_batch, batch_idx):
        x, y_sentiment, y_hierarchy = train_batch
        sentiment_logits = self.model(**x)

        hierarchy_logits = self.get_hierarchy_logits(sentiment_logits)

        sentiment_loss = self.criterion(sentiment_logits, y_sentiment)
        hierarchy_loss = self.criterion(hierarchy_logits, y_hierarchy)

        train_loss = self.alpha_loss*sentiment_loss + (1-self.alpha_loss)*hierarchy_loss

        self.log('train_loss', train_loss)
        self.log('train_sentiment_loss', sentiment_loss)
        self.log('train_hierarchy_loss', hierarchy_loss)

        train_sentiment_metrics = self.train_sentiment_metrics(sentiment_logits, y_sentiment)
        train_hierarchy_metrics = self.train_hierarchy_metrics(hierarchy_logits, y_hierarchy)

        self.log_dict(train_sentiment_metrics, on_step=True, on_epoch=True)
        self.log_dict(train_hierarchy_metrics, on_step=True, on_epoch=True)

        return train_loss


    def validation_step(self, val_batch, batch_idx):
        x, y_sentiment, y_hierarchy = val_batch
        sentiment_logits = self.model(**x)

        hierarchy_logits = self.get_hierarchy_logits(sentiment_logits)

        sentiment_loss = self.criterion(sentiment_logits, y_sentiment)
        hierarchy_loss = self.criterion(hierarchy_logits, y_hierarchy)

        val_loss = self.alpha_loss*sentiment_loss + (1-self.alpha_loss)*hierarchy_loss

        self.log('val_loss', val_loss)
        self.log('val_sentiment_loss', sentiment_loss)
        self.log('train_hierarchy_loss', hierarchy_loss)

        val_sentiment_metrics = self.valid_sentiment_metrics(sentiment_logits, y_sentiment)
        val_hierarchy_metrics = self.valid_hierarchy_metrics(hierarchy_logits, y_hierarchy)

        self.log_dict(val_sentiment_metrics, on_step=True, on_epoch=True)
        self.log_dict(val_hierarchy_metrics, on_step=True, on_epoch=True)


    def forward(self, x):
        sentiment_logits = self.model(**x)
        hierarchy_logits = self.get_hierarchy_logits(sentiment_logits)
        return sentiment_logits, hierarchy_logits


    def test_step(self, test_batch, batch_idx):
        x, y_sentiment, y_hierarchy = test_batch
        sentiment_logits = self.model(**x)
        
        hierarchy_logits = self.get_hierarchy_logits(sentiment_logits)

        sentiment_loss = self.criterion(sentiment_logits, y_sentiment)
        hierarchy_loss = self.criterion(hierarchy_logits, y_hierarchy)

        test_loss = self.alpha_loss*sentiment_loss + (1-self.alpha_loss)*hierarchy_loss

        self.log('test_loss', test_loss)
        self.log('test_sentiment_loss', sentiment_loss)
        self.log('train_hierarchy_loss', hierarchy_loss)

        test_sentiment_metrics = self.test_sentiment_metrics(sentiment_logits, y_sentiment)
        test_hierarchy_metrics = self.test_hierarchy_metrics(hierarchy_logits, y_hierarchy)

        self.log_dict(test_sentiment_metrics, on_step=False, on_epoch=True)
        self.log_dict(test_hierarchy_metrics, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )

        stepping_batches = self.trainer.estimated_stepping_batches

        scheduler = CosineWarmupLR(
            optimizer,
            lr_min=1.0e-6,
            lr_max=self.lr,
            warmup=1200,
            T_max=stepping_batches
        )

        print(f"Warmup Steps: {stepping_batches*self.warmup_ratio}", stepping_batches)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_training_steps=stepping_batches,
        #     num_warmup_steps=stepping_batches*self.warmup_ratio
        # )

        self.trainer.logger.experiment.config["scheduler"] = scheduler.__class__.__name__
        self.trainer.logger.experiment.config["optimizer"] = optimizer.__class__.__name__

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }