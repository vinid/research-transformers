from transformers import Trainer

def default_trainer():
    trainer = Trainer(model_init=self.model_init,
                      args=training_args,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset['test']
                      )
