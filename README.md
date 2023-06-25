# Language-Modelling-and-Smoothing-techniques

## Description 

This repository provides an implementation of N-gram language modeling along with two popular smoothing techniques: Kneser-Kney and Witten Bell. It also includes an in-house tokenizer for preprocessing text data. Additionally, the repository implements a neural model using LSTM architecture and performs perplexity calculations for a comprehensive comparison between the language model and neural model.

## Features

- **N-gram Language Modeling**: The repository supports N-gram language modeling, allowing users to train and generate text based on historical data.
- **Smoothing Techniques**: Two commonly used smoothing techniques, *Kneser-Kney* and *Witten Bell*, are implemented to handle unseen and infrequent N-grams for more accurate probability estimation.
- **Tokenizer**: The repository includes a custom tokenizer that preprocesses text data by splitting it into tokens, enabling efficient language modeling.
- **Neural Model with LSTM**: The repository provides an LSTM-based neural model for language modeling. It learns the patterns and dependencies in text data using deep learning techniques.
- **Perplexity Calculation**: The repository calculates perplexity scores for both the language model and neural model. Perplexity serves as a metric to evaluate the models' performance in predicting word sequences.


## Dataset

The following corpora is used for training and evaluation:

1. Pride and Prejudice Corpus (1,24,970 words): This corpus contains the text from the novel "Pride and Prejudice" by Jane Austen. It comprises approximately 124,970 words.

2. Ulysses Corpus (2,68,117 words): This corpus contains the text from the novel "Ulysses" by James Joyce. It comprises approximately 268,117 words.

Please download the corpus files from this [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ekansh_chauhan_research_iiit_ac_in/EmFusD8Au8hPv8sFXsxHLI8BAu_KKDEUy_geHnZHb5digg?e=FEWfY8)



## File structure

- models/
  - prideModel.h5
  - ulyssus.h5
- language_model.py
- neural_model.py

Here's a breakdown of the file structure:

- `models/`: This folder contains the saved models.
  - `model1.pth`: This file represents a saved model.
  - `model2.pth`: This file represents another saved model.

- `code/`: This folder contains the Python code files.
  - `script1.py`: This file represents a Python script.
  - `script2.py`: This file represents another Python script.


## Running instructions

### language_model.py

To run the language model using N-grams and smoothing techniques, use the following command:

```bash
$ python3 language_model.py <smoothing type> <path to corpus>
```
- <smoothing type>: Specify the smoothing type. Use 'k' for Kneser-Ney or 'w' for Witten-Bell.
- <path to corpus>: Provide the path to the corpus file.

Ex:
```bash
$ python3 language_model.py k ./corpus.txt
```

### neural_model.py

To run the neural model, use the following command:

```bash
$ python3 neural_model.py
```


## Language Model using N-Grams

A language model based on N-grams is a statistical model that predicts the probability of a word or sequence of words based on the previous N-1 words in a text. N-grams represent contiguous sequences of N words in the text.

The formula for calculating the probability of an N-gram is as follows:
```shell
P(w_n | w_1, w_2, ..., w_{n-1}) = Count(w_1, w_2, ..., w_n) / Count(w_1, w_2, ..., w_{n-1})
```

Here,

``` P(w_n | w_1, w_2, ..., w_{n-1})``` :represents the probability of word w_n given the context of previous words w_1, w_2, ..., w_{n-1}. 

```Count(w_1, w_2, ..., w_n)``` denotes the number of occurrences of the N-gram (w_1, w_2, ..., w_n) in the training data, and 

``` Count(w_1, w_2, ..., w_{n-1})``` represents the number of occurrences of the context (w_1, w_2, ..., w_{n-1}).

To estimate the probabilities, we calculate the relative frequencies of N-grams in a large corpus of training data.

## Smoothing in Language Modeling

Smoothing is a technique used in language modeling to handle unseen or infrequent N-grams and improve the accuracy of probability estimates. Kneser-Kney (KK) and Witten Bell (WB) are two commonly used smoothing methods.

### Kneser-Kney Smoothing

Kneser-Kney smoothing addresses the issue of estimating the probability of unseen N-grams while considering their context. It involves discounting and redistributing probabilities. The formula for Kneser-Kney smoothed probability is:

```shell
P_KK(w_n | w_1, w_2, ..., w_{n-1}) = (max(Count(w_1, w_2, ..., w_n) - delta, 0) / Count(w_1, w_2, ..., w_{n-1}))
                                    + (gamma(w_1, w_2, ..., w_{n-1}) / Count(w_1, w_2, ..., w_{n-1})) * P_KK(w_n | w_2, ..., w_{n-1})
```

Here:
- Count(w_1, w_2, ..., w_n): Count of the N-gram (w_1, w_2, ..., w_n)
- Count(w_1, w_2, ..., w_{n-1}): Count of the context (w_1, w_2, ..., w_{n-1})
- delta: A discounting constant
- gamma(w_1, w_2, ..., w_{n-1}): Number of unique N-grams that follow the context (w_1, w_2, ..., w_{n-1})

### Witten Bell Smoothing

Witten Bell smoothing estimates probabilities of unseen N-grams based on the distribution of unseen N-grams. It assigns a separate probability mass to unseen N-grams using a discounted probability mass. The formula for Witten Bell smoothed probability is:

```shell 
P_WB(w_n | w_1, w_2, ..., w_{n-1}) = (gamma(w_1, w_2, ..., w_{n-1}) / (gamma(w_1, w_2, ..., w_{n-1}) + Count(w_1, w_2, ..., w_{n-1}))) * P_WB(w_n)
                                     + (Count(w_1, w_2, ..., w_{n-1}) / (gamma(w_1, w_2, ..., w_{n-1}) + Count(w_1, w_2, ..., w_{n-1}))) * P_ML(w_n)
 ```                                    

Here:
- gamma(w_1, w_2, ..., w_{n-1}): Number of unique N-grams that follow the context (w_1, w_2, ..., w_{n-1})
- Count(w_1, w_2, ..., w_{n-1}): Count of the context (w_1, w_2, ..., w_{n-1})
- P_WB(w_n): Probability of the word w_n estimated from the unseen N-gram distribution
- P_ML(w_n): Maximum likelihood estimate of the word w_n

These smoothing techniques help address the sparsity problem in language modeling and improve the accuracy of probability estimates for both seen and unseen N-grams.

## Perplexity as a Metric for Language Modeling

Perplexity is a widely used evaluation metric for language models. It measures how well a language model predicts a given sequence of words based on its estimated probability distribution.

### Formula for Perplexity

The perplexity (PP) of a language model is calculated using the following formula:
```shell
PP = exp(-sum(log(P(w_n | w_1, w_2, ..., w_{n-1}))) / N)
```

Here:
- PP: Perplexity
- P(w_n | w_1, w_2, ..., w_{n-1}): Probability of word w_n given the context w_1, w_2, ..., w_{n-1}
- log(): Natural logarithm
- sum(): Summation
- N: Total number of words in the test set

Perplexity measures how well a language model predicts a given sequence of words. A lower perplexity indicates better performance, as it means the model assigns higher probabilities to the observed words.

### Interpretation of Perplexity

Perplexity can be interpreted as the average number of equally likely choices the model has for predicting the next word in a sequence. A lower perplexity suggests that the model is more certain about the next word and has a better understanding of the language.

### Using Perplexity for Model Comparison

Perplexity is commonly used to compare different language models. A lower perplexity value indicates a better-performing model in terms of language modeling. It helps in selecting the most suitable model for a specific task or dataset.

## Neural Models vs. Language Models

Neural models and language models are two different approaches used in natural language processing tasks, including language modeling. Let's understand the distinction between the two:

### Language Models

Language models are statistical models that assign probabilities to sequences of words in a language. They are primarily based on the analysis of large amounts of text data. Language models capture the statistical patterns and dependencies in the text to estimate the likelihood of different word sequences. N-gram models and smoothing techniques are commonly used in language modeling.

### Neural Models

Neural models, on the other hand, utilize neural networks, such as recurrent neural networks (RNNs) or transformers, to learn the patterns and structure of language directly from data. These models use distributed representations of words and sequential processing to capture complex relationships and dependencies within text. LSTM (Long Short-Term Memory) and Transformer models are popular choices for neural language models.

### Differences and Advantages

- **Representation:** Language models often rely on simpler representations such as N-grams and frequency-based statistics, while neural models use distributed representations that capture more nuanced information about word meanings and relationships.

- **Modeling Capacity:** Neural models have higher modeling capacity and can capture long-range dependencies and contextual nuances better than traditional language models.

- **Training Process:** Language models are trained using statistical techniques and may require manual feature engineering and smoothing methods. Neural models, on the other hand, can automatically learn representations and features from raw data through end-to-end training.

- **Performance:** Neural models tend to achieve better performance in tasks like language generation, machine translation, and text classification due to their ability to capture complex patterns and context.

- **Data Requirements:** Neural models usually require more training data compared to language models to effectively learn the language's patterns and nuances.

- **Flexibility:** Neural models can be adapted to various tasks beyond language modeling, such as sentiment analysis, named entity recognition, and text summarization.

Both language models and neural models have their own strengths and weaknesses, and the choice between them depends on the specific task, available data, and desired performance.


## Contributions

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

