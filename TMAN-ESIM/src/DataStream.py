import numpy as np
import re

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
        out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val

def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val


def read_all_instances(inpath, word_vocab=None, label_vocab=None, max_sent_length=100, isLower=True):
    instances = []
    infile = open(inpath, 'rt', encoding='utf-8')
    idx = -1
    for line in infile:
        # if(idx == 1000):
        #     break
        idx += 1
        line = line.strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = items[1].strip()
        sentence2 = items[2].strip()
        language = items[3]
        if language == 'True' or language == 'False':
            language = -1
        else:
            language = int(language)
        cur_ID = "{}".format(idx)
        if len(items)>=4: cur_ID = items[3]
        if isLower:
            sentence1 = sentence1.lower()
            sentence2 = sentence2.lower()
        if label_vocab is not None:
            label_id = label_vocab.getIndex(label)
            if label_id >= label_vocab.vocab_size: label_id = 0
        else:
            label_id = int(label)
        word_idx_1 = word_vocab.to_index_sequence(sentence1)
        word_idx_2 = word_vocab.to_index_sequence(sentence2)

        if len(word_idx_1) > max_sent_length:
            word_idx_1 = word_idx_1[:max_sent_length]
        else:
            word_idx_1 = word_idx_1
        if len(word_idx_2) > max_sent_length:
            word_idx_2 = word_idx_2[:max_sent_length]
        instances.append((label, language, sentence1, sentence2, label_id, word_idx_1, word_idx_2, cur_ID))
    infile.close()
    return instances

class DataStream(object):
    def __init__(self, inpath, word_vocab=None, label_vocab=None,
                 isShuffle=False, isLoop=False, isSort=True, options=None):
        instances = read_all_instances(inpath, word_vocab=word_vocab, label_vocab=label_vocab,
                    max_sent_length=options.max_sent_length, isLower=options.isLower)

        # sort instances based on sentence length
        if isSort: instances = sorted(instances, key=lambda instance: (len(instance[5]), len(instance[6]))) # sort instances based on length
        self.num_instances = len(instances)
        
        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, options.batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in range(batch_start, batch_end):
                cur_instances.append(instances[i])
            cur_batch = InstanceBatch(cur_instances)
            self.batches.append(cur_batch)

        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0
    
    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array) 
#         print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def shuffle(self):
        if self.isShuffle: np.random.shuffle(self.index_array)

    def reset(self):
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[self.index_array[i]]


class InstanceBatch(object):
    def __init__(self, instances, with_char=False):
        self.instances = instances
        self.batch_size = len(instances)
        self.question_len = 0
        self.passage_len = 0

        self.question_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_question_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
        self.passage_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_passage_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
        self.label_truth = []  # [batch_size]
        self.id = []
        self.language = []

        for (label, language, sentence1, sentence2, label_id, word_idx_1, word_idx_2, cur_ID) in instances:
            self.id.append(cur_ID)
            cur_question_length = len(word_idx_1)
            cur_passage_length = len(word_idx_2)
            if self.question_len < cur_question_length: self.question_len = cur_question_length
            if self.passage_len < cur_passage_length: self.passage_len = cur_passage_length
            self.question_lengths.append(cur_question_length)
            self.in_question_words.append(word_idx_1)
            self.passage_lengths.append(cur_passage_length)
            self.in_passage_words.append(word_idx_2)
            self.label_truth.append(label_id)
            self.language.append(language)

        # padding all value into np arrays
        self.question_lengths = np.array(self.question_lengths, dtype=np.int32)
        self.in_question_words = pad_2d_vals(self.in_question_words, self.batch_size, self.question_len, dtype=np.int32)
        self.passage_lengths = np.array(self.passage_lengths, dtype=np.int32)
        self.in_passage_words = pad_2d_vals(self.in_passage_words, self.batch_size, self.passage_len, dtype=np.int32)
        self.label_truth = np.array(self.label_truth, dtype=np.int32)
