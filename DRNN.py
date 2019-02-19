import tensorflow as tf


class Model():
    def __init__(self):
        self.word_embedding_dim = 200
        self.pos_embedding_dim = 50
        self.gr_embedding_dim = 50

        self.word_voca_size = 400001
        self.pos_voca_size = 10
        self.gr_voca_size = 21

        self.relation_classes = 19
        self.word_embedding_state_size = 200
        self.pos_embedding_state_size = 50
        self.gr_embedding_state_size = 50
        self.batch_size = 10

        self.channels = 3
        self.lambda_l2 = 0.0001
        self.max_len_path = 10
        self.build_model()

    def build_model(self):
        with tf.name_scope("input"):
            self.path_length = tf.placeholder(tf.int32, shape=[2, self.batch_size], name="path_length")
            self.word_ids = tf.placeholder(tf.int32, shape=[2, self.batch_size, self.max_len_path], name="word_ids")
            self.pos_ids = tf.placeholder(tf.int32, shape=[2, self.batch_size, self.max_len_path], name="pos_ids")
            self.gr_ids = tf.placeholder(tf.int32, shape=[2, self.batch_size, self.max_len_path], name="gr_ids")

            self.y = tf.placeholder(tf.int32, [self.batch_size], name="y")

        with tf.name_scope("word_embedding"):
            self.W_word = tf.Variable(tf.constant(0.1, shape=[self.word_voca_size, self.word_embedding_dim]), name="W_word")
            self.word_embedding_placeholder = tf.placeholder(tf.float32, [self.word_voca_size, self.word_embedding_dim])
            self.word_embedding_init = self.W_word.assign(self.word_embedding_placeholder)
            self.embedded_word = tf.nn.embedding_lookup(self.W_word, self.word_ids)
            self.word_embedding_saver = tf.train.Saver({"word_embedding/W_word": self.W_word})

        with tf.name_scope("pos_embedding"):
            self.W_pos = tf.Variable(tf.random_uniform([self.pos_voca_size, self.pos_embedding_dim]), name="W_pos")
            self.embedded_pos = tf.nn.embedding_lookup(self.W_pos, self.pos_ids)
            self.pos_embedding_saver = tf.train.Saver({"pos_embedding/W_pos": self.W_pos})

        with tf.name_scope("gr_embedding"):
            self.W_gr = tf.Variable(tf.random_uniform([self.gr_voca_size, self.gr_embedding_dim]), name="W_gr")
            self.embedded_gr = tf.nn.embedding_lookup(self.W_gr, self.gr_ids)
            self.gr_embedding_saver = tf.train.Saver({"gr_embedding/W_gr": self.W_gr})

        self.word_hidden_state = tf.zeros([self.batch_size, self.word_embedding_state_size], name="word_hidden_state")
        self.word_cell_state = tf.zeros([self.batch_size, self.word_embedding_state_size], name="word_cell_state")
        self.other_hidden_states = tf.zeros([2, self.batch_size, 50], name="other_hidden_states")
        self.other_cell_states = tf.zeros([2, self.batch_size, 50], name="other_cell_states")

        init_states = [tf.contrib.rnn.LSTMStateTuple(self.other_hidden_states[i], self.other_cell_states[i]) for i in range(2)]
        word_init_state = tf.contrib.rnn.LSTMStateTuple(self.word_hidden_state, self.word_cell_state)

        with tf.variable_scope("word_lstm1"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.word_embedding_state_size)
            state_series, current_state = tf.nn.dynamic_rnn(cell, self.embedded_word[0], sequence_length=self.path_length[0], initial_state=word_init_state)
            self.state_series_word1 = tf.reduce_max(state_series, axis=1)

        with tf.variable_scope("word_lstm2"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.word_embedding_state_size)
            state_series, current_state = tf.nn.dynamic_rnn(cell, self.embedded_word[1], sequence_length=self.path_length[1], initial_state=word_init_state)
            self.state_series_word2 = tf.reduce_max(state_series, axis=1)

        with tf.variable_scope("pos_lstm1"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.pos_embedding_state_size)
            state_series, current_state = tf.nn.dynamic_rnn(cell, self.embedded_pos[0], sequence_length=self.path_length[0], initial_state=init_states[0])
            self.state_series_pos1 = tf.reduce_max(state_series, axis=1)

        with tf.variable_scope("pos_lstm2"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.pos_embedding_state_size)
            state_series, current_state = tf.nn.dynamic_rnn(cell, self.embedded_pos[1], sequence_length=self.path_length[1], initial_state=init_states[0])
            self.state_series_pos2 = tf.reduce_max(state_series, axis=1)

        with tf.variable_scope("gr_lstm1"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.gr_embedding_state_size)
            state_series, current_state = tf.nn.dynamic_rnn(cell, self.embedded_gr[0], sequence_length=self.path_length[0], initial_state=init_states[1])
            self.state_series_gr1 = tf.reduce_max(state_series, axis=1)

        with tf.variable_scope("gr_lstm2"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.gr_embedding_state_size)
            state_series, current_state = tf.nn.dynamic_rnn(cell, self.embedded_gr[1], sequence_length=self.path_length[1], initial_state=init_states[1])
            self.state_series_gr2 = tf.reduce_max(state_series, axis=1)

        self.state_series1 = tf.concat([self.state_series_word1, self.state_series_pos1, self.state_series_gr1], 1)
        self.state_series2 = tf.concat([self.state_series_word2, self.state_series_pos2, self.state_series_gr2], 1)

        self.state_series = tf.concat([self.state_series1, self.state_series2], 1)

        with tf.name_scope("hidden_layer"):
            W_h = tf.Variable(tf.truncated_normal([600, 100], -0.1, 0.1), name="W_h")
            b_h = tf.Variable(tf.zeros([100]), name="b_h")
            y_h = tf.matmul(self.state_series, W_h) + b_h

        with tf.name_scope("softmax_layer"):
            W_s = tf.Variable(tf.truncated_normal([100, self.relation_classes], -0.1, 0.1), name="W_s")
            b_s = tf.Variable(tf.zeros([self.relation_classes]), name="b_s")
            self.logits = tf.matmul(y_h, W_s) + b_s
            self.predictions = tf.argmax(tf.nn.softmax(self.logits, name="predictions"), 1)

        tv_all = tf.trainable_variables()
        tv_regu = []
        non_reg = ["word_embedding/W_word:0", "pos_embedding/W_pos:0", "gr_embedding/W_gr:0", "global_step:0", "hidden_layer/b_h:0", "softmax_layer/b_s:0"]
        for t in tv_all:
            if t.name not in non_reg:
                if t.name.find('biases') == -1:
                    tv_regu.append(t)

        with tf.name_scope("loss"):
            l2_loss = self.lambda_l2*tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.total_loss = loss + l2_loss

        self.global_step = tf.Variable(0, name="global_step")

        with tf.variable_scope("opt"):
            self.opt = tf.train.AdamOptimizer(0.001).minimize(self.total_loss, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

if __name__ == "__main__":
    model = Model()