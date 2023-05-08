# import necessary classes
from fine_tune import FineTune
from transformers import pipeline
from data_preparation import PrepareData

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """fine_tuned_model = FineTune("dbmdz/gpt2-tr-uncased")
    train_acc = fine_tuned_model.train()

    model = fine_tuned_model.get_model()
    tokenizer = fine_tuned_model.get_tokenizer()

    generator = pipeline('text-generation',
                         model=model,
                         tokenizer=tokenizer
                         )  # Specify the GPT model we fine-tuned and give the tokenizer
    # prompt = input('Please give the context of the poem: ')
    # poem = generator(prompt, max_length= 60, do_sample=True, temperature=0.7)
    poem = generator("")
    print(poem[0]['generated_text'])"""

    fine_tuned_model = FineTune("redrussianarmy/gpt2-turkish-cased")
    train_acc = fine_tuned_model.train()
