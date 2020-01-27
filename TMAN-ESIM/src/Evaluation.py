# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import sys
from vocab_utils import Vocab
import namespace_utils
import numpy as np
from sklearn import metrics
import tensorflow as tf
from DataStream import DataStream
from Model import Model

def evaluation(sess, valid_graph, dataStream, name=None, save=None, epoch=None, options=None):
    total = 0
    correct = 0
    label = []
    y_pred = []
    logits = []
    id = []
    for batch_index in range(dataStream.get_num_batch()):  # for each batch
        cur_batch = dataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=False)
        [probs, predictions] = sess.run([valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        for i in range(len(cur_batch.label_truth)):
            label.append(cur_batch.label_truth[i])
            y_pred.append(predictions[i])
            logits.append(probs[i])
            id.append(cur_batch.id[i])
    acc = metrics.accuracy_score(label, y_pred)
    print(name + '_Accuracy: %.4f' % acc)

    if(save):
        write_result(np.array(y_pred), id, options.model_dir + '/../result_target/result' + str(epoch) + '.txt')
        write_logits(np.array(logits), options.model_dir + '/../logits_target/logits' + str(epoch) + '.npy')

    return acc

def write_result(predictions, id, filepath):
    fw = open(filepath, mode='w', encoding='utf-8')
    fw.write("test_id,result\n")
    for i in range(predictions.shape[0]):
        fw.write(str(id[i]) + "," + str(predictions[i]) + "\n")
    fw.close()

def write_logits(logits, filepath):
    with open(filepath, mode='w') as fw:
        for i in range(logits.shape[0]):
            fw.write(str(logits[i][1]) + '\n')
    fw.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=False, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()
    
    # load the configuration file
    print('Loading configurations.')
    options = namespace_utils.load_namespace(args.model_prefix + "ESIM.xnli.config.json")

    if args.word_vec_path is None: args.word_vec_path = options.word_vec_path


    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(args.word_vec_path, fileformat='txt3')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))

    print('Build DataStream ... ')
    testDataStream = DataStream(args.in_path, word_vocab=word_vocab,
                                            label_vocab=None, isShuffle=False, isLoop=True, isSort=True, options=options)
    print('Number of instances in devDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()

    best_path = args.model_prefix + "ESIM.xnli.best.model"
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = Model(3, word_vocab=word_vocab, is_training=False, options=options)

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        print("Restoring model from " + best_path)
        saver.restore(sess, best_path)
        print("DONE!")
        acc = evaluation(sess, valid_graph, testDataStream, epoch=2, save=True, name='Test', options=options)
        print("Accuracy for test set is %.2f" % acc)


