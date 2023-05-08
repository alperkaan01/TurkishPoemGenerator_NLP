from transformers import TrainingArguments

GLOBAL_CONFIG = {
    "training_args":    TrainingArguments(
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
}


class Config(object):
    def __init__(self):
        self.config = GLOBAL_CONFIG

    def get_property(self, property_name: str):
        if property_name not in self.config.keys():
            return None
        return self.config[property_name]