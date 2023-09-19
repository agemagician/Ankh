import os
import random
from functools import partial

import numpy as np
import torch
import wandb
from datasets import load_dataset
from sklearn import metrics
from torch import nn
from torch.utils import data
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

import ankh

seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["WANDB_PROJECT"] = "ankh_solubility_sweeps"

wandb.login()


class SolubilityConvBertModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        input_dim = self.backbone.config.d_model
        self.downstream_model = ankh.ConvBertForBinaryClassification(
            input_dim=input_dim,
            hidden_dim=input_dim // 2,
            kernel_size=7,
            nhead=4,
            num_hidden_layers=1,
            num_layers=1,
            pooling="max",
            dropout=0.1,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        out = self.downstream_model(embeddings, labels)
        return out


class SolubilityDataset(data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, idx):
        return self.ds["sequences"][idx], self.ds["labels"][idx]


def collate_fn(tokenizer):
    def _collate_fn(batch):
        sequences = [example[0] for example in batch]
        labels = [example[1] for example in batch]
        input_ids = tokenizer(
            sequences, padding="longest", return_tensors="pt"
        )
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(
            -1
        )  # (batch_size, 1)
        input_ids.update({"labels": labels})  # in-place update
        return input_ids

    return _collate_fn


def compute_metrics(p: EvalPrediction):
    preds = (torch.sigmoid(torch.tensor(p.predictions)).numpy() > 0.5).tolist()
    labels = p.label_ids.tolist()
    return {
        "accuracy": metrics.accuracy_score(labels, preds),
        "precision": metrics.precision_score(labels, preds),
        "recall": metrics.recall_score(labels, preds),
        "f1": metrics.f1_score(labels, preds),
    }


def initialize_wandb_sweep():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval/loss", "goal": "minimize"},
    }

    parameters_dict = {
        "epochs": {"value": 5},
        "batch_size": {"values": [2, 4]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "gradient_accumulation_steps": {"values": [8, 16, 32]},
        "weight_decay": {"values": [0.0, 0.01, 0.05, 0.1, 0.15]},
    }

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="ankh_solubility_sweeps")
    return sweep_id


def train(
    model, training_dataset, validation_dataset, data_collator, config=None
):
    with wandb.init(config=config):
        config = wandb.config

        training_args = TrainingArguments(
            output_dir="ankh-solubility-sweeps-session",
            report_to="wandb",
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            save_strategy="epoch",
            seed=seed,
            data_seed=seed,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            fp16=False,
            logging_steps=200,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        trainer.train()


def main():
    dataset = load_dataset("proteinea/solubility")
    solubility_train = dataset["train"]
    solubility_validation = dataset["validation"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.device(device):
        base_model, tokenizer = ankh.load_base_model()

        # In case not enough GPU for finetuning Ankh.
        base_model.gradient_checkpointing_enable()

    model = SolubilityConvBertModel(base_model)

    training_dataset = SolubilityDataset(solubility_train)
    validation_dataset = SolubilityDataset(solubility_validation)

    data_collator = collate_fn(tokenizer)

    # `wandb.agent` expects a function that takes `config`,
    # so we create a new partial function with fixed arguments.
    train_fn = partial(
        train, model, training_dataset, validation_dataset, data_collator
    )

    sweep_id = initialize_wandb_sweep()
    wandb.agent(sweep_id, train_fn, count=10)


def solubility_inference(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    sequence: str,
    threshold: float = 0.5,
):
    """
    Inference example for solubility.

    Args:
        model (nn.Module): Task specific model that has Ankh as the backbone
                           and downstream head for the target task,
                           in this case the target task is
                           binary classification.
        tokenizer (AutoTokenizer): Ankh Tokenizer.
        sequence (str): Input sequence that will be used as input to the model
        threshold (float): Threshold for specifying whether the sequence is soluble or not.

        >>> base_model, tokenizer = ankh.load_base_model()
        >>> model = SolubilityConvBertModel(base_model)
        >>> example_sequence = "MEQQMXMLLLMQM"
        >>> threshold = 0.5
        >>> soluble = solubility_inference(model, tokenizer, example_sequence, threshold)
        >>> print(soluble)
    """

    encoded_sequence = tokenizer.encode(sequence)
    model.eval()
    logits = model(encoded_sequence).logits
    soluble = torch.sigmoid(torch.tensor(logits)) > threshold
    return soluble


if __name__ == "__main__":
    main()
