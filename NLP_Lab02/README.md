# Text Data Processing for Neural Network Input

## Cleaning and Preprocessing Steps

The `preprocess_text` function contains the following preprocessing steps:
- Remove mentions and hashtags.
- Remove URLs.
- Remove emojis, symbols, and non-ASCII characters.
- Lowercase the text.
- Handle contractions using `contractions.fix()`.
- Remove punctuation using `text.translate(str.maketrans('', '', string.punctuation))`.
- Remove digits using `re.sub(r'\d+', '', text)`.
- Remove whitespace using `re.sub(r'\s+', ' ', text).strip()`.

The `stem_text` function applies stemming to the text using the Porter Stemmer from NLTK:
- Tokenize the text into words.
- Apply stemming to each word.
- Join the stemmed words back into a single string.

## `process_text_data` Function Summary

This function processes the text data for input to a neural network model.

### Parameters:
- `transformed_X_train` (list): A list of strings, where each string represents a document or sentence.

### Returns:
- `padded_sequences` (numpy.ndarray): A 2D numpy array representing the padded sequences of word indices.
- `word2idx` (dict): A dictionary mapping words to their corresponding indices.
