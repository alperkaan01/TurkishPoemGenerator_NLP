import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel


class PrepareData(object):

    def __init__(self, csv_name):
        self.df = pd.read_csv(csv_name)
        self.dataset = Dataset.from_pandas(self.df)

    def train_test_split(self):
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        return train_df, test_df


