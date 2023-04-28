from datasets import load_dataset


class PrepareData:

    def __init__(self, csv_name):
        self.dataset = load_dataset('csv', data_files=csv_name, delimiter=',')

    def get_dataset(self):
        return self.dataset