import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import pickle
import yaml

np.random.seed(7)

#number of dimensions in input tensor over the vocab size
#1 in this case which represents the blank character
extra_vocab = 1

def filter_and_encode(word, vocab_size, min_len, char_to_id):
    """
    checks if word length is greater than threshold and returns one-hot encoded array along with character sets
    :param word: word string
    :param vocab_size: size of vocabulary (26 in this case)
    :param min_len: word with length less than this is not added to the dataset
    :param char_to_id
    """

    #don't consider words of lengths below a threshold
    word = word.strip().lower()
    if len(word) < min_len:
        return None, None, None

    encoding = np.zeros((len(word), vocab_size + extra_vocab))
    #dict which stores the location at which characters are present
    #e.g. for 'hello', chars = {'h':[0], 'e':[1], 'l':[2,3], 'o':[4]}
    chars = {k: [] for k in range(vocab_size)}

    for i, c in enumerate(word):
        idx = char_to_id[c]
        #update chars dict
        chars[idx].append(i)
        #one-hot encode
        encoding[i][idx] = 1

    return encoding, [x for x in chars.values() if len(x)], set(list(word))


def batchify_words(batch, vocab_size, using_embedding):
    """
    converts a list of words into a batch by padding them to a fixed length array
    :param batch: a list of words encoded using filter_and_encode function
    :param: size of vocabulary (26 in our case)
    :param: use_embedding: if True, 
    """

    total_seq = len(batch)
    if using_embedding:
        #word is a list of indices e.g. 'abd' will be [0,1,3]
        max_len = max([len(x) for x in batch])
        final_batch = []

        for word in batch:
            if max_len != len(word):
                #for index = vocab_size, the embedding gives a 0s vector
                zero_vec = vocab_size*np.ones((max_len - word.shape[0]))
                word = np.concatenate((word, zero_vec), axis=0)
            final_batch.append(word)

        return np.array(final_batch)
    else:
        max_len = max([x.shape[0] for x in batch])
        final_batch = []

        for word in batch:
            #word is a one-hot encoded array of dimensions length x vocab_size
            if max_len != word.shape[0]:
                zero_vec = np.zeros((max_len - word.shape[0], vocab_size + extra_vocab))
                word = np.concatenate((word, zero_vec), axis=0)
            final_batch.append(word)

        return np.array(final_batch)


def encoded_to_string(encoded, target, missed, encoded_len, char_to_id, use_embedding):
    """
    convert an encoded input-output pair back into a string so that we can observe the input into the model
    encoded: array of dimensions padded_word_length x vocab_size
    target: 1 x vocab_size array with 1s at indices wherever character is present
    missed: 1 x vocav_size array with 1s at indices wherever a character which is NOT in the word, is present
    encoded_len: length of word. Needed to retrieve the original word from the padded word
    char_to_id: dict which maps characters to ids
    use_embedding: if character embeddings are used
    """

    #get reverse mapping
    id_to_char = {v:k for k, v in char_to_id.items()}

    if use_embedding:
        word = [id_to_char[x] if x < len(char_to_id) - 1 else '*' for x in list(encoded[:encoded_len])]
    else:
        word = [id_to_char[x] if x < len(char_to_id) - 1 else '*' for x in list(np.argmax(encoded[:encoded_len, :], axis=1))]

    word = ''.join(word)
    target = [id_to_char[x] for x in list(np.where(target != 0)[0])]
    missed = [id_to_char[x] for x in list(np.where(missed != 0)[0])]
    print("Word, target and missed characters:", word, target, missed)



class WordDataset(Dataset):
    def __init__(self, mode, config):
        self.mode = mode
        self.vocab_size = config['vocab_size']
        self.blank_vec = np.zeros((1, self.vocab_size + extra_vocab))
        self.blank_vec[0, self.vocab_size] = 1
        self.cur_epoch = 0
        self.total_epochs = config['epochs']
        
        self.char_to_id = {chr(97+x): x for x in range(self.vocab_size)}
        self.char_to_id['BLANK'] = self.vocab_size
        self.id_to_char = {v:k for k, v in self.char_to_id.items()}
        
        self.drop_uniform = config['drop_uniform']
        self.use_embedding = config['use_embedding']
        self.min_len = config['min_len']
        
        if mode == 'train':
            filename = config['dataset'] + "words_250000_train.txt"
        else:
            filename = config['dataset'] + "words_not_contained.txt"

        pkl_path = config['pickle'] + mode + '_input_dump.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.final_encoded = pickle.load(f)
        else:
            corpus = []
            with open(filename, 'r') as f:
                corpus = f.readlines()

            self.final_encoded = []
            for i, word in enumerate(corpus):
                encoding, unique_pos, chars = filter_and_encode(word, self.vocab_size, self.min_len, self.char_to_id)
                if encoding is not None:
                    self.final_encoded.append((encoding, unique_pos, chars))

            with open(pkl_path, 'wb') as f:
                pickle.dump(self.final_encoded, f)

        print("Length of " + mode + " dataset:", len(self.final_encoded))

    def update_epoch(self, epoch):
        self.cur_epoch = epoch

    def __len__(self):
        return len(self.final_encoded)

    def __getitem__(self, idx):
        word, unique_pos, chars = self.final_encoded[idx]
        
        all_chars = list(self.char_to_id.keys())
        all_chars.remove('BLANK')
        all_chars = set(all_chars)

        drop_prob = 1/(1+np.exp(-self.cur_epoch/self.total_epochs))
        num_to_drop = np.random.binomial(len(unique_pos), drop_prob)
        if num_to_drop == 0:
            num_to_drop = 1

        if self.drop_uniform:
            to_drop = np.random.choice(len(unique_pos), num_to_drop, replace=False)
        else:
            prob = [1/len(x) for x in unique_pos]
            prob_norm = [x/sum(prob) for x in prob]
            to_drop = np.random.choice(len(unique_pos), num_to_drop, p=prob_norm, replace=False)

        drop_idx = []
        for char_group in to_drop:
            drop_idx += unique_pos[char_group]
        
        target = np.clip(np.sum(word[drop_idx], axis=0), 0, 1)
        assert(target[self.vocab_size] == 0)
        target = target[:-1]
        
        input_vec = np.copy(word) 
        input_vec[drop_idx] = self.blank_vec
        
        if self.use_embedding:
            input_vec = np.argmax(input_vec, axis=1)
        
        not_present = np.array(sorted(list(all_chars - chars)))
        num_misses = np.random.randint(0, 10)
        miss_chars = np.random.choice(not_present, num_misses)
        miss_chars = list(set([self.char_to_id[x] for x in miss_chars]))
        
        miss_vec = np.zeros((self.vocab_size))
        miss_vec[miss_chars] = 1
        
        return input_vec, target, miss_vec


class WordDataLoader(DataLoader):
    def __init__(self, mode, config):
        self.dataset = WordDataset(mode, config)

        collate_fn = lambda batch: WordDataLoader.collate_fn(batch, config['vocab_size'], config['use_embedding'])
        super(WordDataLoader, self).__init__(self.dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], collate_fn=collate_fn)

    def update_dataset(self, epoch):
        self.dataset.update_epoch(epoch)
    
    @staticmethod
    def collate_fn(batch, vocab_size, use_embedding):
        lens = np.array([len(x[0]) for x in batch]) 
        inputs = batchify_words([x[0] for x in batch], vocab_size, use_embedding)
        labels = np.array([x[1] for x in batch])
        miss_chars = np.array([x[2] for x in batch])
        return inputs, labels, miss_chars, lens

if __name__ == '__main__':
    with open("deeplearning/config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    train_loader = WordDataLoader('train', config)

    for epoch in range(config['epochs']):
        train_loader.update_dataset(epoch)
        for inputs, labels, miss_chars, lens in train_loader:
            # your training code
            print(inputs, labels, miss_chars, lens)
            exit()