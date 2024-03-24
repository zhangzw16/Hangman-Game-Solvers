# Hangman Game Solvers

## TL;DR
*The overall success rate of 1000 online test words is **0.627***.

TODO

## Hangman game description
<!-- describe hangman game -->

Hangman is a classic word-guessing game that challenges players to deduce a hidden word by guessing one letter at a time. The game begins with the presentation of a series of blanks, each representing a letter of the secret word. Players take turns guessing letters they believe might be in the word. If a guessed letter is correct, it is revealed in its correct positions within the word. If the letter is not in the word, a part of a 'hangman' stick figure is drawn. The objective is to guess the word before the drawing of the hangman is completed, typically 6 incorrect guesses (head, body, 2 arms, 2 legs).

```
* [Game: ] start with a word: _ _ _ _ _
* [Player: ] guess a letter: a
* [Game: ] bingo! now: _ a _ _ _
* [Player: ] guess a letter: e
* [Game: ] sorry, no e in the word: _ a _ _ _
* [Player: ] guess a letter: l
* [Game: ] bingo! now: _ a l l _
* ...... until the word is guessed or the hangman is dead
```

## Preliminary Work

### Local Testing Setup

To enable faster testing and iterative development, I first rewrote the provided API code to create a local version of the Hangman game. To simulate the real test scenario, I obtained a comprehensive English word list from https://github.com/dwyl/english-words. I removed the training set words from this list to create a test set.

Running the baseline algorithm against this test set yielded an accuracy of around 20%, indicating that my constructed dataset has a similar distribution to the real test set and can be used as a validation set.


## Solutions

### Rule-based Strategy 

> The baseline algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words (with the same length) in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.

Digging into the baseline method, I noticed that it only searches for words with exactly matching lengths when performing the search. This ignores many key prefix and suffix matches, such as "apple" to "appleberry".

I tried relaxing the search conditions to match against words of all lengths (`re.match` to `re.search`). This improved the accuracy from 20% to 38.0% (but the searching time also increases).

### N-gram Strategy
Now we try to convert the Hangman game into a mathematical probalistic problem.

First we define the objective as selecting a letter $c$ that is not in the set of already guessed letters $G_t$ and maximizes the probability of guessing the target word $w^*$ given the current game state $w_t$. Mathematically, this can be represented as:

$$C_{t+1} = \arg\max_{c \notin G_t} P(w^*|G_t, w_t, c)$$

To solve this problem, we can use Bayes' theorem to convert it into the problem of computing $P(G_t, w_t, c|w^*)$:

$$P(w^*|G_t, w_t, c) = \frac{P(G_t, w_t, c|w^*)P(w^*)}{P(G_t, w_t, c)}$$

Here, $P(w^*)$ is the prior probability of the target word $w^*$ occurring, which is usually assumed to be uniformly distributed; $P(G_t, w_t, c)$ is the marginal probability, which usually does not need to be computed directly.

The key is to compute $P(G_t, w_t, c|w^*)$, which can be further decomposed into $P(c|G_t, w_t, w^*)$ and $P(G_t, w_t|w^*)$. Among them, $P(c|G_t, w_t, w^*)$ represents the conditional probability that guessing the letter $c$ is correct given the target word $w^*$, the current game state $w_t$, and the set of previously guessed letters $G_t$. This is the part that we approximate using the n-gram model.

The application of the n-gram model is as follows:

- **1-gram**: Directly compute the probability of the letter $c$ appearing in the target word $w^*$.
- **2-gram**: If $w_t$ is in the form of ".x" or "x.", compute the probability of the letter $c$ appearing in the corresponding position of the 2-gram.
- **3-gram**: If $w_t$ is in the form of ".xx", "x.x", or "xx.", compute the probability of the letter $c$ appearing in the corresponding position of the 3-gram.
- **4-gram**: If $w_t$ is in the form of ".xxx", "x.xx", "xx.x", or "xxx.", compute the probability of the letter $c$ appearing in the corresponding position of the 4-gram.

By converting the counts of 1-gram to 4-gram into probability distributions and using preset weights $\alpha$ for weighted summation, we can obtain the final probability of each letter:

$$P(c|w_t) = \sum_{i=1}^{4} \alpha_i P_i(c|w_t)$$

where $P_i(c|w_t)$ is the probability estimate based on the i-gram, and $\alpha_i$ is the weight coefficient.

Finally, we choose the letter $c$ that maximizes $P(w^*|G_t, w_t, c)$ as the next guess. This method makes simplifications in probability, assuming that the probability distribution of the target word can be approximated by the n-gram model, and that the probability of each word being selected is equal. In this way, the n-gram model provides a statistics-based optimization strategy for the Hangman game.

### Neural Network Strategy

#### Words to Machine Learning Samples
- The task is modeled as a multi-class classification problem.
- In this part, words are randomly masked with probability p, and then converted into inputs, labels, miss_chars, and lens. 
  - inputs: 0-26 encoding of the word
  - labels: one-hot encoding of the missing letter
  - miss_chars: randomly generated currently incorrectly predicted letters

#### Model Architecture
In addition to n-gram language models, I also tried using RNNs to model this problem. The model inputs are:
- The current word (represented as a one-hot vector) 
- A miss_chars vector (indicating whether each letter is missing in the word)

The output is a probability distribution over the 26 letters. The specific model structure is as follows:
1. Pass the input one-hot vector through an embedding layer to reduce dimensionality
2. Input the embedded vector into an RNN (I used bidirectional GRU) to extract contextual information from the word
3. Output the hidden state of the last time step of the RNN and reshape it to batch_size x hidden_dim
4. Project the miss_chars vector to a higher dimension through a linear layer
5. Concatenate the RNN output and the high-dimensional representation of miss_chars, then pass through two MLP+ReLU layers, and finally output the probability distribution over the 26 letters

This model can achieve around 60% accuracy on the validation set, which is close to the performance of the 4-gram method. I believe the advantage of the RNN model is that it can model longer-distance dependencies and capture more semantic information. However, the disadvantages are high computational overhead, high model complexity, and the need for more training data and parameter tuning.

#### Training Strategy
- As the number of training rounds increases, gradually increase the probability p to make training more difficult.

## Results and Discussion

| Model |  Score (local) | time (min)| note|
|-------|--------------|--------|----|
|baseline| 21.04 | 1:30 |  |
|re.search| 38.16 | 3:00 | search all words |
|first order| 38.04 | 1:33 | inspired by [this blog](http://www.datagenetics.com/blog/april12012/index.html) |
|N-gram | 62.28 | 0:01 | inspired by [this repo](https://github.com/mattgalarneau/Hangman-NLP), with first order |
|N-gram + 2/4 | 64.32 | 0:07 | with first order |
|N-gram + 2/4 + 2/5  | 66.16 | 0:12 |with first order |
|N-gram + 2/4 + 2/5 + 3/5 | 65.90 | 2:18 |with first order, slow so drop 3/5 |
|GRU-2-1024 | 48.88 | 0:40 | train&test on single NVIDIA-3090 |
|GRU-2-1024 + interval-1| 64.90 | 0:40 | `interval-1` means resample every epoch |
|GRU-4-1024 + interval-1| 69.36 | 0:40 | grid search for optimal structure |
|***GRU + N-gram*** &#10004;| ***73.06*** | 0:43 | boosting the GRU model with N-gram |

## Conclusion and Future Work

## References

codes:
- https://github.com/mattgalarneau/Hangman-NLP
- https://github.com/Jisheng-Liang/hangman_transformer
- https://github.com/methi1999/hangman?tab=readme-ov-file
- https://github.com/Azure/Hangman/blob/master/Train%20a%20Neural%20Network%20to%20Play%20Hangman.ipynb

others:
- https://stackoverflow.com/questions/9942861/optimal-algorithm-for-winning-hangman
- https://blog.wolfram.com/2010/08/13/25-best-hangman-words/
- http://www.datagenetics.com/blog/april12012/index.html
- https://blog.csdn.net/weixin_42327556/article/details/103285869