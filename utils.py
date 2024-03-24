import collections
import re
import numpy as np

LETTER_ORDER_DICT = {
    1: ['a', 'i'],
    2: ['a', 'o', 'e', 'i', 'u', 'm', 'b', 'h'],
    3: ['a', 'e', 'o', 'i', 'u', 'y', 'h', 'b', 'c', 'k'],
    4: ['a', 'e', 'o', 'i', 'u', 'y', 's', 'b', 'f'],
    5: ['s', 'e', 'a', 'o', 'i', 'u', 'y', 'h'],
    6: ['e', 'a', 'i', 'o', 'u', 's', 'y'],
    7: ['e', 'i', 'a', 'o', 'u', 's'],
    8: ['e', 'i', 'a', 'o', 'u'],
    9: ['e', 'i', 'a', 'o', 'u'],
    10: ['e', 'i', 'o', 'a', 'u'],
    11: ['e', 'i', 'o', 'a', 'd'],
    12: ['e', 'i', 'o', 'a', 'f'],
    13: ['i', 'e', 'o', 'a'],
    14: ['i', 'e', 'o'],
    15: ['i', 'e', 'a'],
    16: ['i', 'e', 'h'],
    17: ['i', 'e', 'r'],
    18: ['i', 'e', 'a'],
    19: ['i', 'e', 'a'],
    20: ['i', 'e']
}

def build_n_gram(word_list, max_n):
    # create n-gram from word list
    n_grams = {}
    for n in range(1, max_n + 1):
        n_grams[n] = collections.defaultdict(int)
        for word in word_list:
            for i in range(len(word) - n + 1):
                n_grams[n][word[i:i + n]] += 1
    return n_grams


def build_n_gram_from_file(file_path, max_n=5):
    # create n-gram from file
    with open(file_path, 'r') as f:
        word_list = f.read().splitlines()
    return build_n_gram(word_list, max_n)


def get_n_gram_prob(n_grams, word, guessed_letters):
    # gen range(26) and exclude guessed_letters
    not_guessed_letters = [i for i in range(26) if chr(97 + i) not in guessed_letters]
    # get probability of next letter from n-gram
    next_letter_count = np.zeros(26, dtype=float)
    alphas = [0.05, 0.1, 0.2, 0.3, 0.5]
    # add counts of 1-gram
    gram_1_count = np.array([n_grams[1][chr(97 + j)] if j in not_guessed_letters else 0 for j in range(26)])
    next_letter_count += alphas[0] * (gram_1_count / sum(gram_1_count))
    # add counts of 2-gram
    for i in range(len(word) - 1):
        # if word[i:i+2] fits: .x or x., x means any alphabet, add count
        gram_2_count = np.zeros(26)
        if word[i] == '.' and word[i+1] != '.':
            gram_2_count = np.array([n_grams[2][chr(97 + j) + word[i+1]] if j in not_guessed_letters else 0 for j in range(26)])
        elif word[i] != '.' and word[i+1] == '.':
            gram_2_count = np.array([n_grams[2][word[i] + chr(97 + j)] if j in not_guessed_letters else 0 for j in range(26)])
        if sum(gram_2_count) != 0:
            next_letter_count += alphas[1] * (gram_2_count / sum(gram_2_count))
    # add counts of 3-gram
    for i in range(len(word) - 2):
        gram_3_count = np.zeros(26)
        # if word[i:i+3] fits: .xx or x.x or xx., x means any alphabet, add count
        if word[i] == '.' and word[i+1] != '.' and word[i+2] != '.':
            gram_3_count = np.array([n_grams[3][chr(97 + j) + word[i+1:i+3]] if j in not_guessed_letters else 0 for j in range(26)])
        elif word[i] != '.' and word[i+1] == '.' and word[i+2] != '.':
            gram_3_count = np.array([n_grams[3][word[i] + chr(97 + j) + word[i+2]] if j in not_guessed_letters else 0 for j in range(26)])
        elif word[i] != '.' and word[i+1] != '.' and word[i+2] == '.':
            gram_3_count = np.array([n_grams[3][word[i:i+2] + chr(97 + j)] if j in not_guessed_letters else 0 for j in range(26)])
        if sum(gram_3_count) != 0:
            next_letter_count += alphas[2] * (gram_3_count / sum(gram_3_count))
    # add counts of 4-gram
    for i in range(len(word) - 3):
        gram_4_count = np.zeros(26)
        # if word[i:i+4] fits: .xxx or x.xx or xx.x or xxx., x means any alphabet, add count
        if word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.':
            gram_4_count = np.array([n_grams[4][chr(97 + j) + word[i+1:i+4]] if j in not_guessed_letters else 0 for j in range(26)])
        elif word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.':
            gram_4_count = np.array([n_grams[4][word[i] + chr(97 + j) + word[i+2:i+4]] if j in not_guessed_letters else 0 for j in range(26)])
        elif word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.':
            gram_4_count = np.array([n_grams[4][word[i:i+2] + chr(97 + j) + word[i+3]] if j in not_guessed_letters else 0 for j in range(26)])
        elif word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.':
            gram_4_count = np.array([n_grams[4][word[i:i+3] + chr(97 + j)] if j in not_guessed_letters else 0 for j in range(26)])
        
        if sum(gram_4_count) != 0:
            next_letter_count += alphas[3] * (gram_4_count / sum(gram_4_count))

        gram_4_2_count = np.zeros(26)
        # if word[i:i+4] fits: two . and two x, add count
        if word[i] == '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.':
            for j in range(26):
                for k in range(26):
                    if j in not_guessed_letters and k in not_guessed_letters:
                        gram_4_2_count[j] += n_grams[4][chr(97 + j) + chr(97 + k) + word[i+2:i+4]]
                        gram_4_2_count[k] += n_grams[4][chr(97 + j) + chr(97 + k) + word[i+2:i+4]]
        elif word[i] != '.' and word[i+1] == '.' and word[i+2] == '.' and word[i+3] != '.':
            for j in range(26):
                for k in range(26):
                    if j in not_guessed_letters and k in not_guessed_letters:
                        gram_4_2_count[j] += n_grams[4][word[i] + chr(97 + j) + chr(97 + k) + word[i+3]]
                        gram_4_2_count[k] += n_grams[4][word[i] + chr(97 + j) + chr(97 + k) + word[i+3]]
        elif word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] == '.':
            for j in range(26):
                for k in range(26):
                    if j in not_guessed_letters and k in not_guessed_letters:
                        gram_4_2_count[j] += n_grams[4][word[i:i+2] + chr(97 + j) + chr(97 + k)]
                        gram_4_2_count[k] += n_grams[4][word[i:i+2] + chr(97 + j) + chr(97 + k)]
        elif word[i] == '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.':
            for j in range(26):
                for k in range(26):
                    if j in not_guessed_letters and k in not_guessed_letters:
                        gram_4_2_count[j] += n_grams[4][chr(97 + j) + word[i+1] + chr(97 + k) + word[i+3]]
                        gram_4_2_count[k] += n_grams[4][chr(97 + j) + word[i+1] + chr(97 + k) + word[i+3]]
        elif word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] == '.':
            for j in range(26):
                for k in range(26):
                    if j in not_guessed_letters and k in not_guessed_letters:
                        gram_4_2_count[j] += n_grams[4][word[i] + chr(97 + j) + word[i+2] + chr(97 + k)]
                        gram_4_2_count[k] += n_grams[4][word[i] + chr(97 + j) + word[i+2] + chr(97 + k)]
        elif word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.':
            for j in range(26):
                for k in range(26):
                    if j in not_guessed_letters and k in not_guessed_letters:
                        gram_4_2_count[j] += n_grams[4][chr(97 + j) + word[i+1:i+3] + chr(97 + k)]
                        gram_4_2_count[k] += n_grams[4][chr(97 + j) + word[i+1:i+3] + chr(97 + k)]
        
        if sum(gram_4_2_count) != 0:
            next_letter_count += (alphas[3] / 2) * (gram_4_2_count / sum(gram_4_2_count))

    # add counts of 5-gram
    for i in range(len(word) - 4):
        dot_count = sum([1 for c in word[i:i+5] if c == '.'])
        
        if dot_count == 1:
            gram_5_count = np.zeros(26)
            # if word[i:i+5] fits: .xxxx or x.xxx or xx.xx or xxx.x or xxxx., x means any alphabet, add count
            if word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.':
                gram_5_count = np.array([n_grams[5][chr(97 + j) + word[i+1:i+5]] if j in not_guessed_letters else 0 for j in range(26)])
            elif word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.':
                gram_5_count = np.array([n_grams[5][word[i] + chr(97 + j) + word[i+2:i+5]] if j in not_guessed_letters else 0 for j in range(26)])
            elif word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] != '.':
                gram_5_count = np.array([n_grams[5][word[i:i+2] + chr(97 + j) + word[i+3:i+5]] if j in not_guessed_letters else 0 for j in range(26)])
            elif word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] != '.':
                gram_5_count = np.array([n_grams[5][word[i:i+3] + chr(97 + j) + word[i+4]] if j in not_guessed_letters else 0 for j in range(26)])
            elif word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] == '.':
                gram_5_count = np.array([n_grams[5][word[i:i+4] + chr(97 + j)] if j in not_guessed_letters else 0 for j in range(26)])
            if sum(gram_5_count) != 0:
                next_letter_count += alphas[4] * (gram_5_count / sum(gram_5_count))

        elif dot_count == 2:
            gram_5_2_count = np.zeros(26)
            # if word[i:i+5] fits: two . and three x, add count
            if word[i] == '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][chr(97 + j) + chr(97 + k) + word[i+2:i+5]]
                            gram_5_2_count[k] += n_grams[5][chr(97 + j) + chr(97 + k) + word[i+2:i+5]]
            elif word[i] != '.' and word[i+1] == '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] != '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][word[i] + chr(97 + j) + chr(97 + k) + word[i+3:i+5]]
                            gram_5_2_count[k] += n_grams[5][word[i] + chr(97 + j) + chr(97 + k) + word[i+3:i+5]]
            elif word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] == '.' and word[i+4] != '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][word[i:i+2] + chr(97 + j) + chr(97 + k) + word[i+4]]
                            gram_5_2_count[k] += n_grams[5][word[i:i+2] + chr(97 + j) + chr(97 + k) + word[i+4]]
            elif word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] == '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][word[i:i+3] + chr(97 + j) + chr(97 + k)]
                            gram_5_2_count[k] += n_grams[5][word[i:i+3] + chr(97 + j) + chr(97 + k)]
            elif word[i] == '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] != '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][chr(97 + j) + word[i+1] + chr(97 + k) + word[i+3:i+5]]
                            gram_5_2_count[k] += n_grams[5][chr(97 + j) + word[i+1] + chr(97 + k) + word[i+3:i+5]]
            elif word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] != '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][chr(97 + j) + word[i+1:i+3] + chr(97 + k) + word[i+4]]
                            gram_5_2_count[k] += n_grams[5][chr(97 + j) + word[i+1:i+3] + chr(97 + k) + word[i+4]]
            elif word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] == '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][chr(97 + j) + word[i+1:i+4] + chr(97 + k)]
                            gram_5_2_count[k] += n_grams[5][chr(97 + j) + word[i+1:i+4] + chr(97 + k)]
            elif word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] != '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][word[i] + chr(97 + j) + word[i+2] + chr(97 + k) + word[i+4]]
                            gram_5_2_count[k] += n_grams[5][word[i] + chr(97 + j) + word[i+2] + chr(97 + k) + word[i+4]]
            elif word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] == '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][word[i] + chr(97 + j) + word[i+2:i+4] + chr(97 + k)]
                            gram_5_2_count[k] += n_grams[5][word[i] + chr(97 + j) + word[i+2:i+4] + chr(97 + k)]
            elif word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] == '.':
                for j in range(26):
                    for k in range(26):
                        if j in not_guessed_letters and k in not_guessed_letters:
                            gram_5_2_count[j] += n_grams[5][word[i:i+2] + chr(97 + j) + word[i+3] + chr(97 + k)]
                            gram_5_2_count[k] += n_grams[5][word[i:i+2] + chr(97 + j) + word[i+3] + chr(97 + k)]
            
            if sum(gram_5_2_count) != 0:
                next_letter_count += (alphas[4] / 2) * (gram_5_2_count / sum(gram_5_2_count))

        # elif dot_count == 3:
        #     gram_5_3_count = np.zeros(26)
        #     # if word[i:i+5] fits: three . and two x, add count
        #     # xx...
        #     if word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] == '.' and word[i+4] == '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][word[i:i+2] + chr(97 + j) + chr(97 + k) + chr(97 + l)]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # .xx..
        #     elif word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] == '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][chr(97 + j) + word[i+1:i+3] + chr(97 + k) + chr(97 + l)]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # ..xx.
        #     elif word[i] == '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] == '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][chr(97 + j) + chr(97 + k) + word[i+2:i+4] + chr(97 + l)]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # ...xx
        #     elif word[i] == '.' and word[i+1] == '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] != '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][chr(97 + j) + chr(97 + k) + chr(97 + l) + word[i+3:i+5]]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # x.x..
        #     elif word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] == '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][word[i] + chr(97 + j) + word[i+2] + chr(97 + k) + chr(97 + l)]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # x..x.
        #     elif word[i] != '.' and word[i+1] == '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] == '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][word[i] + chr(97 + j) + chr(97 + k) + word[i+3] + chr(97 + l)]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # x...x
        #     elif word[i] != '.' and word[i+1] == '.' and word[i+2] == '.' and word[i+3] == '.' and word[i+4] != '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][word[i] + chr(97 + j) + chr(97 + k) + chr(97 + l) + word[i+4]]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # .x.x.
        #     elif word[i] == '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] == '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][chr(97 + j) + word[i+1] + chr(97 + k) + word[i+3] + chr(97 + l)]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # .x..x
        #     elif word[i] == '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] == '.' and word[i+4] != '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][chr(97 + j) + word[i+1] + chr(97 + k) + chr(97 + l) + word[i+4]]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
        #     # ..x.x
        #     elif word[i] == '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] != '.':
        #         for j in range(26):
        #             for k in range(26):
        #                 for l in range(26):
        #                     if j in not_guessed_letters and k in not_guessed_letters and l in not_guessed_letters:
        #                         res = n_grams[5][chr(97 + j) + chr(97 + k) + word[i+2] + chr(97 + l) + word[i+4]]
        #                         gram_5_3_count[j] += res
        #                         gram_5_3_count[k] += res
        #                         gram_5_3_count[l] += res
            
        #     if sum(gram_5_3_count) != 0:
        #         next_letter_count += (alphas[4] / 3) * (gram_5_3_count / sum(gram_5_3_count))
        
        #normalize
        next_letter_count /= sum(next_letter_count)

    return next_letter_count
    
## old guess
# def guess(self, word): # word input example: "_ p p _ e "
#     ###############################################
#     # Replace with your own "guess" function here #
#     ###############################################

#     # clean the word so that we strip away the space characters
#     # replace "_" with "." as "." indicates any character in regular expressions
#     clean_word = word[::2].replace("_",".")
#     len_right_letters = len(clean_word) - clean_word.count('.')
    
#     # find length of passed word
#     len_word = len(clean_word)
    
#     # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
#     # self.current_dictionary = 
#     new_dictionary = []

#     if len_right_letters == 0 and len_word in LETTER_ORDER_DICT:
#         order = LETTER_ORDER_DICT[len_word]
#         for letter in order:
#             if letter not in self.guessed_letters:
#                 return letter

#     # iterate through all of the words in the old plausible dictionary
#     for dict_word in self.current_dictionary:
#         # continue if the word is not of the appropriate length
#         # if len(dict_word) != len_word:
#         #     continue
            
#         # if dictionary word is a possible match then add it to the current dictionary
#         search = re.search(clean_word,dict_word)
#         if search:
#             # if len_right_letters == 3:
#             #     print(search.group(0), dict_word, clean_word)
#             new_dictionary.append(search.group(0))
        
    
#     # overwrite old possible words dictionary with updated version
#     self.current_dictionary = new_dictionary
    
    
#     # count occurrence of all characters in possible word matches
#     full_dict_string = "".join(self.current_dictionary)
    
#     c = collections.Counter(full_dict_string)
#     sorted_letter_count = c.most_common()                   
    
#     guess_letter = '!'
    
#     # return most frequently occurring letter in all possible words that hasn't been guessed yet
#     for letter,instance_count in sorted_letter_count:
#         if letter not in self.guessed_letters:
#             guess_letter = letter
#             break


    
#     # if no word matches in training dictionary, default back to ordering of full dictionary
#     if guess_letter == '!':
#         sorted_letter_count = self.full_dictionary_common_letter_sorted
#         for letter,instance_count in sorted_letter_count:
#             if letter not in self.guessed_letters:
#                 guess_letter = letter
#                 break            
    
#     return guess_letter