import tensorflow as tf
import layer_utils

regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

class Model(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, options=None, global_step=None):
        self.dropout_rate = 0.0
        if (is_training):
            options.dropout_rate = options.dropout_rate
        self.options = options
        self.create_placeholders()
        self.create_model_graph(num_classes, word_vocab, char_vocab, is_training, global_step=global_step)

    def create_placeholders(self):
        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.language = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.in_question_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
        self.in_passage_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]

    def create_feed_dict(self, cur_batch, is_training=False):
        feed_dict = {
            self.question_lengths: cur_batch.question_lengths,
            self.passage_lengths: cur_batch.passage_lengths,
            self.in_question_words: cur_batch.in_question_words,
            self.in_passage_words: cur_batch.in_passage_words,
            self.truth: cur_batch.label_truth,
            self.language: cur_batch.language,
        }
        return feed_dict

    def create_model_graph(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, global_step=None):
        options = self.options
        # ======word representation layer======
        with tf.variable_scope("Input_Embedding_Layer"):
            if word_vocab is not None:
                word_vec_trainable = True
                cur_device = '/gpu:0'
                if options.fix_word_vec:
                    word_vec_trainable = False
                    cur_device = '/cpu:0'
                with tf.device(cur_device):
                    self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                                                          initializer=tf.constant(word_vocab.word_vecs),
                                                          dtype=tf.float32)
        c_emb = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words)
        q_emb = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words)
        if is_training:
            c_emb = tf.nn.dropout(c_emb, 1 - self.dropout_rate)
            q_emb = tf.nn.dropout(q_emb, 1 - self.dropout_rate)
        input_shape = tf.shape(self.in_question_words)
        question_lengths = input_shape[1]
        input_shape = tf.shape(self.in_passage_words)
        passage_lengths = input_shape[1]
        c_mask = tf.sequence_mask(self.passage_lengths, passage_lengths, dtype=tf.float32)  # [batch_size, passage_len]
        q_mask = tf.sequence_mask(self.question_lengths, question_lengths, dtype=tf.float32)  # [batch_size, question_len]
        with tf.variable_scope("Encoder", reuse=None):
            match_vec = Matching_Model(c_emb, q_emb, self.passage_lengths, self.question_lengths, c_mask, q_mask,
                                    is_training, self.dropout_rate, options)
            dim = int(match_vec.shape[-1])
        with tf.variable_scope("language_classification", reuse=None):
            logits_adv = tf.layers.dense(inputs=match_vec, units=2, activation=None,
                                             use_bias=True, name='logits_adv', reuse=False)
        with tf.variable_scope("entailment_classification", reuse=None):
            logits = tf.layers.dense(inputs=match_vec, units=3, activation=None,
                                             use_bias=True, name='logits', reuse=False)

        self.prob = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.prob, 1)

        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        language_matrix = tf.one_hot(self.language, 2, dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix) )
        self.loss_d = 0.1 * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits_adv, labels=language_matrix))
        self.loss_g = self.loss - self.loss_d

        if not is_training: return

        theta_lc = []
        theta_ec = []
        theta_g = []

        for var in tf.trainable_variables():
            if var.name.startswith("Model/language_classification"):
                theta_lc.append(var)
            if var.name.startswith("Model/entailment_classification"):
                theta_ec.append(var)
            if var.name.startswith("Model/Encoder"):
                theta_g.append(var)
            print(var.name)

        if self.options.lambda_l2 > 0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in theta_ec if tf.trainable_variables() if not v.name.endswith('embedding')])
            self.loss = self.loss + self.options.lambda_l2 * l2_loss
            l2_loss_d = tf.add_n([tf.nn.l2_loss(v) for v in theta_lc if tf.trainable_variables() if not v.name.endswith('embedding')])
            self.loss_d = self.loss_d + self.options.lambda_l2 * l2_loss_d
            l2_loss_g = tf.add_n([tf.nn.l2_loss(v) for v in theta_g if tf.trainable_variables() if not v.name.endswith('embedding')])
            self.loss_g = self.loss_g + self.options.lambda_l2 * l2_loss_g

        if self.options.optimize_type == 'adadelta':
            self.lc_solove = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate).minimize(self.loss_d,
                                                                                                           var_list=theta_lc)
            self.ec_solove = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate).minimize(self.loss,
                                                                                                           var_list=theta_ec)
            self.g_solove = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate).minimize(self.loss_g,
                                                                                                          var_list=theta_g)
        elif self.options.optimize_type == 'adam':
            self.lc_solove = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate).minimize(self.loss_d,
                                                                                                           var_list=theta_lc)
            self.ec_solove = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate).minimize(self.loss,
                                                                                                           var_list=theta_ec)
            self.g_solove = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate).minimize(self.loss_g,
                                                                                                          var_list=theta_g)

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)

def Matching_Model(c_emb, q_emb, passage_lengths, question_lengths, c_mask, q_mask,
                   is_training, dropout_rate, options):

    with tf.variable_scope("Embedding_Encoder_Layer"):
        q_emb = tf.multiply(q_emb, tf.expand_dims(q_mask, axis=-1))
        c_emb = tf.multiply(c_emb, tf.expand_dims(c_mask, axis=-1))

        (q_fw, q_bw, q) = layer_utils.my_lstm_layer(
            q_emb, options.context_lstm_dim, input_lengths=question_lengths, scope_name="context_represent",
            reuse=False, is_training=is_training, dropout_rate=dropout_rate, use_cudnn=options.use_cudnn)

        (c_fw, c_bw, c) = layer_utils.my_lstm_layer(
            c_emb, options.context_lstm_dim, input_lengths=passage_lengths, scope_name="context_represent",
            reuse=True, is_training=is_training, dropout_rate=dropout_rate, use_cudnn=options.use_cudnn)

        q = tf.multiply(q, tf.expand_dims(q_mask, axis=-1))
        c = tf.multiply(c, tf.expand_dims(c_mask, axis=-1))

    with tf.variable_scope("Co-attention_Layer"):
        c2q, q2c = dot_attention(q, c, q_mask, c_mask)

    with tf.variable_scope("Model_Encoder_Layer"):
        passage_inputs = tf.concat([c2q, c, c2q * c, c - c2q], axis=2)
        question_inputs = tf.concat([q2c, q, q2c * q, q - q2c], axis=2)
        passage_inputs = tf.layers.dense(inputs=passage_inputs, units=2 * options.context_lstm_dim, activation=tf.nn.relu,
                                         use_bias=True, name='pro', reuse=False)
        question_inputs = tf.layers.dense(inputs=question_inputs, units=2 * options.context_lstm_dim, activation=tf.nn.relu,
                                          use_bias=True, name='pro', reuse=True)
        question_inputs = tf.multiply(question_inputs, tf.expand_dims(q_mask, axis=-1))
        passage_inputs = tf.multiply(passage_inputs, tf.expand_dims(c_mask, axis=-1))

        (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
            question_inputs, options.aggregation_lstm_dim, input_lengths=question_lengths,
            scope_name='aggregate_layer',
            reuse=False, is_training=is_training, dropout_rate=dropout_rate, use_cudnn=options.use_cudnn)

        question_inputs = cur_aggregation_representation

        (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
            passage_inputs, options.aggregation_lstm_dim,
            input_lengths=passage_lengths, scope_name='aggregate_layer',
            reuse=True, is_training=is_training, dropout_rate=dropout_rate, use_cudnn=options.use_cudnn)
        passage_inputs = cur_aggregation_representation  # [batch_size, passage_len, 2*aggregation_lstm_dim]
        # if is_training:
        #     question_inputs = tf.nn.dropout(question_inputs, (1 - options.dropout_rate))
        #     passage_inputs = tf.nn.dropout(passage_inputs, (1 - options.dropout_rate))
        question_inputs = tf.multiply(question_inputs, tf.expand_dims(q_mask, axis=-1))
        passage_inputs = tf.multiply(passage_inputs, tf.expand_dims(c_mask, axis=-1))

        passage_outputs_mean = tf.div(tf.reduce_sum(passage_inputs, 1),
                                      tf.expand_dims(tf.cast(passage_lengths, tf.float32), -1))
        question_outputs_mean = tf.div(tf.reduce_sum(question_inputs, 1),
                                       tf.expand_dims(tf.cast(question_lengths, tf.float32), -1))
        passage_outputs_max = tf.reduce_max(passage_inputs, axis=1)
        question_outputs_max = tf.reduce_max(question_inputs, axis=1)
        input_dim = int(passage_inputs.shape[2])

        question_outputs = tf.concat([question_outputs_max, question_outputs_mean], axis=1)
        passage_outputs = tf.concat([passage_outputs_max, passage_outputs_mean], axis=1)
        match_representation = tf.concat(axis=1, values=[question_outputs, passage_outputs])
    # ========Prediction Layer=========
    if is_training: match_representation = tf.nn.dropout(match_representation, (1 - dropout_rate))

    return match_representation


def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    # relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    # relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    relevancy_matrix = mask_logits(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = mask_logits(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def dot_attention(q, c, q_mask, c_mask, scope="Dot_Attention_Layer"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        s = tf.einsum("abd,acd->abc", c, q)
        s = mask_relevancy_matrix(s, tf.cast(q_mask, tf.float32), tf.cast(c_mask, tf.float32))
        s_q = tf.nn.softmax(s, dim=1)
        q2c = tf.einsum("abd,abc->acd", c, s_q)
        s_c = tf.nn.softmax(s, dim=2)
        c2q = tf.einsum("abd,acb->acd", q, s_c)
        return c2q, q2c
