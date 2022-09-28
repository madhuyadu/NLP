import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# Tokenizes whole sentence into words/punctuations/characters/numbers etc
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

# Stems each word into its root form while also converting to lower case
def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

# Create BOW matrix given tokenized sentence and word vocabulary as inputs
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]  ---> tokenised sentence
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]   ----> word vocabulary
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]   ----> BOW representation/matrix
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]

    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):   # iterate through each of the words in the vocabulary
        if w in sentence_words:     # check for existence of word (from vocab) in the tokenised sentence
            bag[idx] = 1          # set the corresponding index/position to 1 when the word exists

    return bag  # return BOW array for a sentence

# CHECKS to see if the functions are working fine
# print(tokenize("Hi, hello how are you?"))
# print([stem(word) for word in ["giving", "gives", "organize", "needless"]])
# print(bag_of_words(["hello", "how", "are", "you"],["hi", "hello", "I", "you", "bye", "thank", "cool"] ))