import numpy as np
import string

alphabet = string.ascii_lowercase


def word_to_array(word):

    word_array = np.zeros((5,26), dtype=bool)

    for pos, letter in enumerate(word):
        word_array[pos][alphabet.find(letter)] = 1

    return(word_array)


def cross_check(candidate_guesses_array, poss_words_array):

    output = np.matmul(np.expand_dims(candidate_guesses_array, 1),
                       np.swapaxes(poss_words_array,1,2))

    return(output)