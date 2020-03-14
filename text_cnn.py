import tensorflow as tf


class TextCNN(object):
    def __init__(self, sequence_length, num_class, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform((vocab_size, embedding_size), -1.0, 1.0, name="W"))
            self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)
        pooled_outs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedding_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID',
                                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name='pool')
            pooled_outs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filters_total, num_class],
                                initializer=tf.contrib.layers.xavier_initializer()
                                )
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="score")
            self.prediction = tf.arg_max(self.scores, 1, name="prediction")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_loss * l2_reg_lambda

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction, tf.arg_max(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
