import argparse
import os
import numpy as np
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory to the dataset.')
    parser.add_argument('--wordvec_path', type=str, required=True, help='Directory to the word vectors.')
    args, unparsed = parser.parse_known_args()
    dataset_dir = args.dataset_dir
    wordvec_path = args.wordvec_path
    id = 4
    vocab = {}
    vocab['_PAD'] = 0
    vocab['_UNK'] = 1
    vocab['_GO'] = 2
    vocab['_EOS'] = 3

    with open(os.path.join(dataset_dir, 'train_enzh.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    with open(os.path.join(dataset_dir, 'dev_en.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    with open(os.path.join(dataset_dir, 'test_en.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    with open(os.path.join(dataset_dir, 'dev_zh.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    with open(os.path.join(dataset_dir, 'test_zh.txt'), mode='r') as fd:
        for line in fd.readlines():
            array = line.lower().split('\t')
            word_list = array[1].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
            word_list = array[2].split(' ')
            for word in word_list:
                if (word not in vocab and word != ''):
                    vocab[word] = id
                    id += 1
    print("vocab has", len(vocab), "entries (not _PAD or _UNK or _GO or _EOS)")
    embeddings = None
    glove_dimensionality = None

    # pass over glove data copying data into embedddings array
    # for the cases where the token is in the reference vocab.
    tokens_set = set(vocab.keys())
    # tokens_requiring_random = set()
    remove_set = set()
    glove_embedding_norms = []
    for line in open(wordvec_path, "r"):
        cols = line.strip().split(" ")
        if (len(cols) < 10):
            continue
        token = cols[0]
        if token in vocab:
            glove_embedding = np.array(cols[1:], dtype=np.float32)
            if embeddings is None:
                glove_dimensionality = len(glove_embedding)
                embeddings = np.empty((len(vocab), glove_dimensionality), dtype=np.float32)  # +1 for pad & unk
            assert len(glove_embedding) == glove_dimensionality, "differing dimensionality in glove data?"
            embeddings[vocab[token]] = glove_embedding
            remove_set.add(token)
            glove_embedding_norms.append(np.linalg.norm(glove_embedding))

    # given these embeddings we can calculate the median norm of the glove data
    tokens_requiring_random = tokens_set - remove_set
    print(tokens_requiring_random)
    print(len(tokens_requiring_random))
    median_glove_embedding_norm = np.median(glove_embedding_norms)

    print(sys.stderr, "build .npy file")
    print(sys.stderr, "after passing over glove there are", len(tokens_requiring_random), "tokens requiring a random alloc")


    # return a random embedding with the same norm as the glove data median norm
    def random_embedding():
        random_embedding = np.random.randn(1, glove_dimensionality)
        random_embedding /= np.linalg.norm(random_embedding)
        random_embedding *= median_glove_embedding_norm
        return random_embedding


    # assign PAD and UNK random embeddings (pre projection)
    embeddings[0] = random_embedding()  # PAD
    embeddings[1] = random_embedding()  # UNK

    # assign random projections for every other fields requiring it
    for token in tokens_requiring_random:
        embeddings[vocab[token]] = random_embedding()

    # p = random_projection.GaussianRandomProjection(n_components=300)
    # embeddings = p.fit_transform(embeddings)

    # zero out PAD embedding
    embeddings[0] = [0] * embeddings.shape[1]

    print(embeddings.shape)
    # write embeddings npy to disk

    vocab_ids = {}
    for word in vocab:
        vocab_ids[vocab[word]] = word

    fw = open(os.path.join(dataset_dir, 'xnli_embedding_enzh.txt'), mode='w')
    for i in range(int(embeddings.shape[0])):
        fw.write(vocab_ids[i])
        for j in range(300):
            fw.write(" " + str(embeddings[i][j]))
        fw.write('\n')
    fw.close()
