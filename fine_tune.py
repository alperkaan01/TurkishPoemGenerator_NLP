# install necessary packages
from transformers \
    import AutoTokenizer, GPT2LMHeadModel, \
    Trainer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TextDataset
from data_preparation import PrepareData
from config import Config
import evaluate
import torch
from torch.utils.data import TensorDataset
from transformers import default_data_collator


def eval_results(trainer, eval_dataset):
    return trainer.evaluate(eval_dataset)


class FineTune:
    def __init__(self, pretrained_model="dbmdz/gpt2-tr-uncased"):
        self.pretrained_model = pretrained_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.model = GPT2LMHeadModel(self.pretrained_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model)

        prep_data = PrepareData('data/turkish_poems_formatted.csv')  # change here with your csv
        self.train_df, self.eval_df = prep_data.train_test_split()
        self.train_df = self.train_df.dropna()
        self.eval_df = self.eval_df.dropna()

        self.train_encodings = self.tokenizer(self.train_df['content'].tolist(), padding=True, truncation=True, max_length=512)
        self.eval_encodings = self.tokenizer(self.eval_df['content'].tolist(), padding=True, truncation=True, max_length=512)
        self.config = Config()

    """def tokenize_function(self, arg):
        return self.tokenizer(arg['text'], padding=True, truncation=True)

    def tokenize_dataset(self, dataset):
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']
        return train_dataset, eval_dataset"""

    def train(self):
        training_args = self.config.get_property("training_args")

        train_matrix_id = torch.tensor(self.train_encodings['input_ids'])
        train_matrix_attention = torch.tensor(self.train_encodings['attention_mask'])
        eval_matrix_id = torch.tensor(self.eval_encodings['input_ids'])
        eval_matrix_attention = torch.tensor(self.eval_encodings['attention_mask'])

        print(train_matrix_id)

        train_dataset = TensorDataset(train_matrix_id,
                                      train_matrix_attention)
        eval_dataset = TensorDataset(eval_matrix_id,
                                     eval_matrix_attention)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
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
