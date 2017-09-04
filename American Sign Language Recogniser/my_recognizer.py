import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    data = test_set.get_all_Xlengths().values()
    for X, lengths in data:
        best_probability, best_word, prob_list = float("-inf"), None, {}
        for word, model in models.items():
            try:
                prob_list[word] = model.score(X, lengths)
            except:
                prob_list[word] = float("-inf")
            if prob_list[word] > best_probability:
                best_probability = prob_list[word]
                best_word = word
        probabilities.append(prob_list)
        guesses.append(best_word)
    return probabilities, guesses





