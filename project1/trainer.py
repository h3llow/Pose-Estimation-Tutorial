from copy import deepcopy

import numpy as np
import sklearn.metrics
import torch
import time

import wandb
from transformers import TrainingArguments, TrainerCallback
from transformers import Trainer, EarlyStoppingCallback
from datasets import load_metric


class PTTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, train_loader, val_loader, epochs=50, lr=0.001, device="cuda", patience=5, log=True):

        criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        train_accuracy = []
        val_accuracy = []
        train_loss = []
        val_loss = []
        self.model.to(device)

        curr_patience = 0

        start_time = time.time()

        # Start loop
        for epoch in range(epochs):  # (loop for every epoch)
            print("Epoch {} running".format(epoch))  # (printing message)
            """ Training Phase """

            self.model.train()
            model_loss = 0
            model_corrects = 0
            for i, (inputs, labels) in enumerate(train_loader):
                # bs, ncrops, c, h, w = inputs.size()
                # inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs.view(-1, c, h, w)/255)
                labels = labels.to(device)
                inputs = inputs.to(device)

                # forward inputs and get output

                # outputs = model(inputs).view(bs, ncrops, -1).mean(1)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # get loss value and update the network weights
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                model_loss += loss.item()
                model_corrects += torch.sum(preds == labels.data).item()
                del loss, outputs

            loss = model_loss / (len(train_loader))
            acc = model_corrects / (len(train_loader) * train_loader.batch_size)
            train_accuracy.append(acc)
            train_loss.append(loss)
            # Print progress
            if log:
                print('Train Acc: {:.2f}% Loss: {:.2f} Time: {:.0f}s'.format(acc * 100, loss, time.time() - start_time))

            """ Validation Phase """
            self.model.eval()
            model_loss = 0
            model_corrects = 0
            for i, (inputs, labels) in enumerate(val_loader):
                with torch.no_grad():
                    # bs, ncrops, c, h, w = inputs.size()
                    # inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs.view(-1, c, h, w)/255)
                    labels = labels.to(device)
                    inputs = inputs.to(device)

                    # forward inputs and get output
                    self.optimizer.zero_grad()
                    # outputs = model(inputs).view(bs, ncrops, -1).mean(1)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # get loss value and update the network weights
                    model_loss += loss.item()
                    model_corrects += torch.sum(preds == labels.data).item()
                    del loss, outputs

            loss = model_loss / (len(val_loader))
            acc = model_corrects / (len(val_loader) * val_loader.batch_size)

            if np.any(loss >= np.array(val_loss)):
                curr_patience += 1
            else:
                curr_patience = 0
                torch.save(self.model, "best_resnet.pth")

            val_accuracy.append(acc)
            val_loss.append(loss)
            if log:
                print('Validation Acc: {:.2f}% Loss: {:.2f} Time: {:.0f}s'.format(acc * 100, loss, time.time() - start_time))
            if curr_patience == patience:
                break
        torch.save(self.model, "last_resnet.pth")
        return train_accuracy, val_accuracy, train_loss, val_loss

    def validate(self, test_loader, device="cuda"):
        model = self.model.to(device)
        model.eval()
        pred = []
        label = []

        start = time.time()
        for i, (inputs, labels) in enumerate(test_loader):
            with torch.no_grad():
                label += list(labels)
                inputs = inputs.to(device)

                # forward inputs and get output
                self.optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                pred += list(preds.to("cpu"))
        end = time.time()
        print("Avergae inference time per image is", str(round(((end - start) * 1000) / len(label))) + "ms on the device " + torch.cuda.get_device_name() + ".")
        return label, pred

class HFTrainer:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def train(self, train_set, val_set, bs=64, epochs=50, lr=2e-4, patience=5, report_to="", checkpoint_dir="x_checkpoints"):
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=bs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            fp16=True,
            save_steps=bs,
            logging_strategy="epoch",
            eval_steps=bs,
            logging_steps=10,
            learning_rate=lr,
            save_total_limit=10,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=report_to,
            load_best_model_at_end=True
        )

        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([x[0] for x in batch]),
                'labels': torch.tensor([x[1] for x in batch])
            }

        metric = load_metric("accuracy")

        def compute_metrics(p):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_set,
            eval_dataset=val_set,
            tokenizer=self.processor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )


        class CustomCallback(TrainerCallback):

            def __init__(self, trainer) -> None:
                super().__init__()
                self._trainer = trainer

            def on_epoch_end(self, args, state, control, **kwargs):
                if control.should_evaluate:
                    control_copy = deepcopy(control)
                    self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                    return control_copy
        trainer.add_callback(CustomCallback(trainer))

        trainer.train()
        wandb.finish()
        metrics = {"train_accuracy": [], "eval_accuracy": [], "loss": [], "eval_loss": []}
        for entry in trainer.state.log_history:
            for metric in ["train_accuracy", "eval_accuracy", "loss", "eval_loss"]:
                if metric in entry:
                    metrics[metric].append(entry[metric])

        return metrics["train_accuracy"], metrics["eval_accuracy"], metrics["loss"], metrics["eval_loss"]
    def validate(self, test_loader, device="cuda"):
        model = self.model.to(device)
        model.eval()
        pred = []
        label = []
        start = time.time()
        for i, (inputs, labels) in enumerate(test_loader):
            with torch.no_grad():
                label += list(labels)
                inputs = inputs.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs.logits, 1)
                pred += list(preds.to("cpu"))

        end = time.time()
        print("Average inference time per image is", str(round(((end - start) * 1000) / len(label))) + "ms on the device " + torch.cuda.get_device_name() + ".")
        return label, pred