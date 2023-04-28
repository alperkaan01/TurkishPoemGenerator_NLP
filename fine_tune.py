# install necessary packages
from transformers \
    import AutoTokenizer, GPT2LMHeadModel, TrainingArguments,\
    Trainer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from data_preparation import PrepareData
import evaluate


def eval_results(trainer, eval_dataset):
    return trainer.evaluate(eval_dataset)


class FineTune:
    def __init__(self, pretrained_model="dbmdz/gpt2-tr-uncased"):
        self.pretrained_model = pretrained_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        #self.model = GPT2LMHeadModel(self.pretrained_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model)

        self.dataset = PrepareData('turkish_poems.csv').get_dataset()

    def tokenize_function(self, arg):
        return self.tokenizer(arg['text'], padding=True, truncation=True)

    def tokenize_dataset(self, dataset):
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']
        return train_dataset, eval_dataset

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            evaluation_strategy='steps',
            save_total_limit=2,
            save_steps=500,
            logging_steps=500,
            learning_rate=2e-5,
            load_best_model_at_end=True
        )

        train_dataset, eval_dataset = self.tokenize_dataset(self.dataset)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        trainer.train()

        accuracy_score = eval_results(trainer, eval_dataset)
        return accuracy_score

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
