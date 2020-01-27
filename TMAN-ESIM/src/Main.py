# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import numpy as np

from Model import Model
from vocab_utils import Vocab
from DataStream import DataStream
import namespace_utils
from sklearn import metrics

def collect_vocabs(train_path, with_POS=False, with_NER=False):
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt', encoding='utf-8')
    for line in infile:
        line = line.strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = re.split("\\s+",items[1].lower())
        sentence2 = re.split("\\s+",items[2].lower())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        if with_POS: 
            all_POSs.update(re.split("\\s+",items[3]))
            all_POSs.update(re.split("\\s+",items[4]))
        if with_NER: 
            all_NERs.update(re.split("\\s+",items[5]))
            all_NERs.update(re.split("\\s+",items[6]))
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def output_probs(probs, label_vocab):
    out_string = ""
    for i in range(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()

def evaluation(sess, valid_graph, dataStream, name=None, save=None, epoch=None, type=None, options=None):
    total = 0
    correct = 0
    label = []
    y_pred = []
    logits = []
    id = []
    for batch_index in range(dataStream.get_num_batch()):  # for each batch
        cur_batch = dataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=True)
        [probs, predictions] = sess.run([valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        for i in range(len(cur_batch.label_truth)):
            label.append(cur_batch.label_truth[i])
            y_pred.append(predictions[i])
            logits.append(probs[i])
            id.append(cur_batch.id[i])
    acc = metrics.accuracy_score(label, y_pred)
    if(save):
        if(name != 'Dev'):
            write_result(np.array(y_pred), id, options.model_dir + '/../result_' + type + '/result' + str(epoch) + '.txt')
            write_multi_logits(np.array(logits), options.model_dir + '/../logits_' + type + '/logits' + str(epoch) + '.npy')
    return acc

def train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, testDataStream, devDataStream_target, testDataStream_target, options, best_path, best_path_target):
    best_accuracy_source_dev = -1
    best_accuracy_target_dev = -1
    best_accuracy_source_test = -1
    best_accuracy_target_test = -1
    best_e_source = -1
    best_e_target = -1
    for epoch in range(options.max_epochs):
        print('Train in epoch %d' % epoch)
        # training
        trainDataStream.shuffle()
        num_batch = trainDataStream.get_num_batch()
        start_time = time.time()
        total_loss = 0
        total_loss_d = 0
        total_loss_g = 0
        true_y = []
        pred_y = []
        for batch_index in range(num_batch):  # for each batch
            cur_batch = trainDataStream.get_batch(batch_index)
            feed_dict = train_graph.create_feed_dict(cur_batch, is_training=True)
            _, loss_d = sess.run([train_graph.lc_solove, train_graph.loss_d], feed_dict=feed_dict)
            _, loss_value = sess.run([train_graph.ec_solove, train_graph.loss], feed_dict=feed_dict)
            _, loss_g, prediction = sess.run([train_graph.g_solove, train_graph.loss_g, train_graph.predictions], feed_dict=feed_dict)
            # print(s)
            total_loss += loss_value
            total_loss_d += loss_d
            total_loss_g += loss_g
            for i in range(len(cur_batch.label_truth)):
                true_y.append(cur_batch.label_truth[i])
                pred_y.append(prediction[i])
            if batch_index % 100 == 0:
                print('{} '.format(batch_index), end="")
                sys.stdout.flush()

        duration = time.time() - start_time
        print('Epoch %d: loss = %.4f (%.3f sec)' % (epoch, total_loss / num_batch, duration))

        print('Epoch %d: loss_d = %.4f (%.3f sec)' % (epoch, total_loss_d / num_batch, duration))

        print('Epoch %d: loss_g = %.4f (%.3f sec)' % (epoch, total_loss_g / num_batch, duration))
        # evaluation
        start_time = time.time()
        print('TRAIN_ACC: %.4f' % metrics.accuracy_score(true_y, pred_y))

        duration = time.time() - start_time
        acc = evaluation(sess, valid_graph, devDataStream, name='Dev', type='source')
        print("SOURCE_DEV_ACC: %.2f" % acc)
        print('Evaluation time for source: %.3f sec' % (duration))
        if acc > best_accuracy_source_dev:
            best_e_source = epoch
            best_accuracy_source_dev = acc
            acc = evaluation(sess, valid_graph, testDataStream, name='Test', epoch=epoch, type='source',
                       options=options)
            print("SOURCE_TEST_ACC: %.2f" % acc)
            best_accuracy_source_test = acc
            saver.save(sess, best_path)

        start_time = time.time()
        acc = evaluation(sess, valid_graph, devDataStream_target, name='Dev', type='target')
        duration = time.time() - start_time
        print("TARGET_DEV_ACC: %.2f" % acc)
        print('Evaluation time for target: %.3f sec' % (duration))
        if acc > best_accuracy_target_dev:
            best_e_target = epoch
            best_accuracy_target_dev = acc
            acc = evaluation(sess, valid_graph, testDataStream_target, name='Test', epoch=epoch, type='target',
                       options=options)
            best_accuracy_target_test = acc
            print("TARGET_TEST_ACC: %.2f" % acc)
            saver.save(sess, best_path_target)
        print("=" * 20 + "BEST_DEV_ACC_SOURCE in epoch(" + str(best_e_source) + "): %.3f" % best_accuracy_source_dev + "=" * 20)
        print("=" * 20 + "BEST_TEST_ACC_SOURCE in epoch(" + str(best_e_source) + "): %.3f" % best_accuracy_source_test + "=" * 20)
        print("=" * 20 + "BEST_DEV_ACC_TARGET in epoch(" + str(best_e_target) + "): %.3f" % best_accuracy_target_dev + "=" * 20)
        print("=" * 20 + "BEST_TEST_ACC_TARGET in epoch(" + str(best_e_target) + "): %.3f" % best_accuracy_target_test + "=" * 20)



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

def write_multi_logits(logits, filename):
    np.save(filename, logits)

def main(FLAGS):
    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    dev_path_target = FLAGS.dev_path_target
    test_path_target = FLAGS.test_path_target
    word_vec_path = FLAGS.word_vec_path
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, '../result_source'))
        os.makedirs(os.path.join(log_dir, '../logits_source'))
        os.makedirs(os.path.join(log_dir, '../result_target'))
        os.makedirs(os.path.join(log_dir, '../logits_target'))

    log_dir_target = FLAGS.model_dir + '_target'
    if not os.path.exists(log_dir_target):
        os.makedirs(log_dir_target)


    path_prefix = log_dir + "/ESIM.{}".format(FLAGS.suffix)
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    path_prefix_target = log_dir_target + "/ESIM.{}".format(FLAGS.suffix)
    namespace_utils.save_namespace(FLAGS, path_prefix_target + ".config.json")
    # build vocabs
    word_vocab = Vocab(word_vec_path, fileformat='txt3')

    best_path = path_prefix + '.best.model'
    best_path_target = path_prefix_target + '.best.model'
    char_path = path_prefix + ".char_vocab"
    label_path = path_prefix + ".label_vocab"
    has_pre_trained_model = False
    char_vocab = None
    # if os.path.exists(best_path + ".index"):
    print('Collecting words, chars and labels ...')
    (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path)
    print('Number of words: {}'.format(len(all_words)))
    label_vocab = Vocab(fileformat='voc', voc=all_labels, dim=2)
    label_vocab.dump_to_txt2(label_path)

    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    print("Number of labels: {}".format(num_classes))
    sys.stdout.flush()


    print('Build SentenceMatchDataStream ... ')
    trainDataStream = DataStream(train_path, word_vocab=word_vocab, label_vocab=None,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    sys.stdout.flush()

    devDataStream = DataStream(dev_path, word_vocab=word_vocab, label_vocab=None,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    sys.stdout.flush()

    testDataStream = DataStream(test_path, word_vocab=word_vocab, label_vocab=None,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)

    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()

    devDataStream_target = DataStream(dev_path_target, word_vocab=word_vocab, label_vocab=None,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    sys.stdout.flush()

    testDataStream_target = DataStream(test_path_target, word_vocab=word_vocab, label_vocab=None,
                                                isShuffle=True, isLoop=True, isSort=True, options=FLAGS)

    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()


    with tf.Graph().as_default():
        initializer = tf.contrib.layers.xavier_initializer()
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = Model(num_classes, word_vocab=word_vocab,
                                                    is_training=True, options=FLAGS, global_step=global_step)
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = Model(num_classes, word_vocab=word_vocab,
                is_training=False, options=FLAGS)

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
#             if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(initializer)

            # training
            train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, testDataStream, devDataStream_target, testDataStream_target, FLAGS, best_path, best_path_target)

def enrich_options(options):
    if "in_format"not in options.__dict__.keys():
        options.__dict__["in_format"] = 'tsv'

    return options

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, help='Path to the test set.')
    parser.add_argument('--dev_path_target', type=str, help='Path to the target language dev set.')
    parser.add_argument('--test_path_target', type=str, help='Path to the target language test set.')
    parser.add_argument('--word_vec_path', type=str, help='Path the to pre-trained word vector model.')

    parser.add_argument('--model_dir', type=str, help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--suffix', type=str, default='normal', help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')

    parser.add_argument('--config_path', type=str, help='Configuration file.')

#     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    args, unparsed = parser.parse_known_args()
    if args.config_path is not None:
        print('Loading the configuration from ' + args.config_path)
        FLAGS = namespace_utils.load_namespace(args.config_path)
    else:
        FLAGS = args
    sys.stdout.flush()

    print(FLAGS.train_path)
    main(FLAGS)

