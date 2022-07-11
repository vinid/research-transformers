import random
from datasets import Dataset
import optuna
from typing import Callable
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

class RepeatedTransformer:

    def __init__(self, model_name, tokenizer_name):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

    def model_init(self):
        return AutoModel.from_pretrained(self.model_name)

    def run(self, training_data, validation_data, test_data, n=5, trainer_parameters=None, training_args_parameters=None):
        predictions = {}

        for run_index in range(0, n):
            if trainer_parameters is not None:
                training_args = TrainingArguments(**training_args_parameters)
                trainer = Trainer(**trainer_parameters, args=training_args,
                                  train_dataset=training_data,
                                  eval_dataset=validation_data)
            else:
                training_args = None
                trainer = Trainer(model_init=self.model_init,
                                  args=training_args,
                                  train_dataset=training_data,
                                  eval_dataset=validation_data)
            trainer.train()

            predictions[run_index] = trainer.predict(test_data)

        return predictions


class OptimalTransformer:

    def __init__(self, model_name, tokenizer_name, num_labels,
                 experiment_name, model_init: Callable, dataset: Dataset, compute_metrics: Callable, parameters: dict, seed=10):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_init = model_init
        self.tokenizer_name = tokenizer_name
        self.experiment_name = experiment_name
        self.dataset = dataset
        self.compute_metrics = compute_metrics
        self.parameters = parameters
        self.seed = seed

    def objective(self, trial: optuna.Trial):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        training_args = TrainingArguments(
            output_dir=f"{self.experiment_name}_{trial.number}",
            learning_rate=trial.suggest_loguniform('learning_rate', low=4e-5, high=0.01),
            weight_decay=trial.suggest_loguniform('weight_decay', 4e-5, 0.01),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            disable_tqdm=False,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy="steps",
            metric_for_best_model='loss',
            load_best_model_at_end=True,
            seed=self.seed)

        trainer = Trainer(model_init=self.model_init,
                          args=training_args,
                          tokenizer=tokenizer,
                          train_dataset=self.dataset["train"],
                          eval_dataset=self.dataset["validation"]
                          )
        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_loss"]

    def run(self, n_trials):
        study = optuna.create_study(study_name='hyper-parameter-search',
                                    direction='minimize')
        study.optimize(func=self.objective, n_trials=n_trials)

        return study.best_params, study.best_value, study.best_trial
