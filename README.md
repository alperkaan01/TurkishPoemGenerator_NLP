#TurkishPoemGenerator_NLP
TurkishPoemGenerator_NLP is a project that demonstrates the use of natural language processing (NLP) techniques to generate Turkish poems using a fine-tuned GPT-2 model.

Project Overview
The project involves collecting a dataset of Turkish poems, pre-processing the data, fine-tuning a GPT-2 model on the pre-processed data, and generating new poems using the fine-tuned model.

The project uses Python and various NLP libraries, including transformers, pandas, and numpy. The transformers library is used to load and fine-tune the GPT-2 model, while pandas and numpy are used for data pre-processing and manipulation.

Getting Started
Prerequisites
To run the project, you'll need the following:

Python 3.x
transformers, pandas, and numpy libraries
Installing
To install the required libraries, run:

Copy code
pip install -r requirements.txt
Usage
To generate new Turkish poems using the pre-trained GPT-2 model, run:

Copy code
python generate_poems.py
The generated poems will be saved to a file named generated_poems.txt.

Dataset
The dataset used for the project is a collection of Turkish poems in CSV format. The dataset is included in the data directory.

Fine-Tuning the Model
To fine-tune the GPT-2 model on the Turkish poems dataset, run:

Copy code
python train.py
This will save the fine-tuned model to the output directory.

Evaluation
The generated poems can be evaluated using perplexity and human evaluation. The implementation of these metrics is included in the evaluation.py file.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This project was inspired by the work of OpenAI and the transformers library.
The Turkish poems dataset used in this project was sourced from various online sources.
