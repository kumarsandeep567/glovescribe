# GloVe Scribe - Neural Text Generation with Semantic Embeddings

## Project Overview
- GloVe Scribe is a text generation project that combines Recurrent Neural Networks (RNNs) with pre-trained GloVe word embeddings to generate meaningful and contextually relevant text. 
- The project demonstrates the implementation of modern NLP techniques using both Keras and TensorFlow frameworks.

## Setup Instructions

- Clone the repository and navigate to it:
   ```bash
   git clone https://github.com/kumarsandeep567/glovescribe.git

   cd glovescribe
   ```

- Please install the dependencies by running 
   ```bash
   pip install -r requirements.txt
   ```

- **Installing TensorFlow** : To ensure you install the correct version of TensorFlow, please refer to TensorFlow's official installation guide [here](https://www.tensorflow.org/install/pip)

## Features
- Text generation using LSTM-based RNN architecture
- Integration of pre-trained GloVe embeddings for semantic understanding
- Customizable sequence length and generation parameters
- Text preprocessing and cleaning utilities
- Support for both sentence-level and paragraph-level generation

## Project Structure
- `GloVe_Scribe.ipynb`: Main Jupyter notebook containing the implementation
- `Heavens and Earth.txt`: Sample training dataset
- `nlp_model.h5`: Trained model file
- `glove.6B.100d.txt`: Pre-trained GloVe embeddings

## Implementation
The project follows a systematic approach to text generation:

1. Text Preprocessing
    - Cleaning and normalization
    - Sentence tokenization
    - Vocabulary building

2. Model Architecture
    - Embedding layer (GloVe)
    - LSTM layer (256 units)
    - Dense output layer with softmax activation

3. Training
    - Sequence length: 3 words
    - Batch size: 64
    - Training/Validation split: 80/20

## Acknowledgments
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014 (Stanford University) - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)