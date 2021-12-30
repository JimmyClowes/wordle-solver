import numpy as np
import string
from collections import Counter
from math import log

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


def binary_encode_cross_check(cross_check_output):

    # get diagonals for exact matches
    diags = np.diagonal(cross_check_output, axis1=2, axis2=3)

    # get rowmaxes excluding diagonal for correct letters in incorrect positions
    rowmaxes = np.logical_and((np.tril(cross_check_output, -1) +
                               np.triu(cross_check_output, 1)).max(axis=3),
                              ~diags)

    # concatenate exact and inexact matches
    diag_rowmax = np.concatenate((diags, rowmaxes), axis=2)

    # encode 10 element boolean arrays as binary numbers
    diag_rowmax_bin_encode = np.matmul(diag_rowmax, np.array([2**i for i in range(9, -1, -1)]))

    diag_rowmax_bin_encode = diag_rowmax_bin_encode.astype('>i2')

    return(diag_rowmax_bin_encode)


def calc_entropy(diag_rowmax_bin_encode, element, log_base=2):

    # Determine number of possible remaining words
    num_poss_remaining = len(diag_rowmax_bin_encode[element])

    # Count frequency of each binary encoded outcome
    counts = Counter(diag_rowmax_bin_encode[element])

    # Calculate expected entropy
    entropy = sum([(tup[1]/num_poss_remaining) * log(1/tup[1],log_base) for tup in list(counts.items())])

    return(entropy)