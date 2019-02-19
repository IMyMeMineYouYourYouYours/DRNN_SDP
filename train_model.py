import _pickle as pickle
import tensorflow as tf
import numpy as np
from DRNN import Model

data_dir = 'C:/Users/Ailab_cho/PycharmProjects/DRNN_SDP/data'
ckpt_dir = 'C:/Users/Ailab_cho/PycharmProjects/DRNN_SDP/checkpoint'
word_embed_dir = 'C:/Users/Ailab_cho/PycharmProjects/DRNN_SDP/checkpoint/word_embed'
model_dir = 'C:/Users/Ailab_cho/PycharmProjects/DRNN_SDP/checkpoint/model'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = Model()

f = open(data_dir + '/vocab.pkl', 'rb')
vocab = pickle.load(f)
f.close()

word2id = dict((w, i) for i,w in enumerate(vocab))
id2word = dict((i, w) for i,w in enumerate(vocab))

unknown_token = "UNKNOWN_TOKEN"
word2id[unknown_token] = model.word_voca_size -1
id2word[model.word_voca_size-1] = unknown_token

pos_tags_vocab = []
for line in open(data_dir + '/pos_tags.txt'):
        pos_tags_vocab.append(line.strip())

dep_vocab = []
for line in open(data_dir + '/dependency_types.txt'):
    dep_vocab.append(line.strip())

relation_vocab = []
for line in open(data_dir + '/relation_types.txt'):
    relation_vocab.append(line.strip())


rel2id = dict((w, i) for i,w in enumerate(relation_vocab))
id2rel = dict((i, w) for i,w in enumerate(relation_vocab))

pos_tag2id = dict((w, i) for i,w in enumerate(pos_tags_vocab))
id2pos_tag = dict((i, w) for i,w in enumerate(pos_tags_vocab))

dep2id = dict((w, i) for i,w in enumerate(dep_vocab))
id2dep = dict((i, w) for i,w in enumerate(dep_vocab))

pos_tag2id['OTH'] = 9
id2pos_tag[9] = 'OTH'

dep2id['OTH'] = 20
id2dep[20] = 'OTH'

JJ_pos_tags = ['JJ', 'JJR', 'JJS']
NN_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
RB_pos_tags = ['RB', 'RBR', 'RBS']
PRP_pos_tags = ['PRP', 'PRP$']
VB_pos_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
_pos_tags = ['CC', 'CD', 'DT', 'IN']

def pos_tag(x):
    if x in JJ_pos_tags:
        return pos_tag2id['JJ']
    if x in NN_pos_tags:
        return pos_tag2id['NN']
    if x in RB_pos_tags:
        return pos_tag2id['RB']
    if x in PRP_pos_tags:
        return pos_tag2id['PRP']
    if x in VB_pos_tags:
        return pos_tag2id['VB']
    if x in _pos_tags:
        return pos_tag2id[x]
    else:
        return 9

f = open(data_dir + '/train_paths', 'rb')
word_p1, word_p2, gr_p1, gr_p2, pos_p1, pos_p2 = pickle.load(f)
f.close()

relations = []
for line in open(data_dir + '/train_relations.txt'):
    relations.append(line.strip())

length = len(word_p1)
num_batches = int(length/model.batch_size)

for i in range(length):
    for j, word in enumerate(word_p1[i]):
        word = word.lower()
        word_p1[i][j] = word if word in word2id else unknown_token
    for k, word in enumerate(word_p2[i]):
        word = word.lower()
        word_p2[i][k] = word if word in word2id else unknown_token
    for l, g in enumerate(gr_p1[i]):
        gr_p1[i][l] = g if g in dep2id else 'OTH'
    for m, g in enumerate(gr_p2[i]):
        gr_p2[i][m] = g if g in dep2id else 'OTH'

word_p1_ids = np.ones([length, model.max_len_path], dtype=int)
word_p2_ids = np.ones([length, model.max_len_path], dtype=int)
pos_p1_ids = np.ones([length, model.max_len_path], dtype=int)
pos_p2_ids = np.ones([length, model.max_len_path], dtype=int)
dep_p1_ids = np.ones([length, model.max_len_path], dtype=int)
dep_p2_ids = np.ones([length, model.max_len_path], dtype=int)
rel_ids = np.array([rel2id[rel] for rel in relations])
path1_len = np.array([len(w) for w in word_p1], dtype=int)
path2_len = np.array([len(w) for w in word_p2])

for i in range(length):
    for j, w in enumerate(word_p1[i]):
        word_p1_ids[i][j] = word2id[w]
    for j, w in enumerate(word_p2[i]):
        word_p2_ids[i][j] = word2id[w]
    for j, w in enumerate(pos_p1[i]):
        pos_p1_ids[i][j] = pos_tag(w)
    for j, w in enumerate(pos_p2[i]):
        pos_p2_ids[i][j] = pos_tag(w)
    for j, w in enumerate(gr_p1[i]):
        dep_p1_ids[i][j] = dep2id[w]
    for j, w in enumerate(gr_p2[i]):
        dep_p2_ids[i][j] = dep2id[w]

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    ckpt = tf.train.latest_checkpoint(model_dir)
    #saver.restore(sess, ckpt)

    num_epochs = 10
    for i in range(num_epochs):
        for j in range(num_batches):
            path_dict = [path1_len[j * model.batch_size:(j + 1) * model.batch_size], path2_len[j * model.batch_size:(j + 1) * model.batch_size]]
            word_dict = [word_p1_ids[j * model.batch_size:(j + 1) * model.batch_size],
                         word_p2_ids[j * model.batch_size:(j + 1) * model.batch_size]]
            pos_dict = [pos_p1_ids[j * model.batch_size:(j + 1) * model.batch_size],
                        pos_p2_ids[j * model.batch_size:(j + 1) * model.batch_size]]
            dep_dict = [dep_p1_ids[j * model.batch_size:(j + 1) * model.batch_size],
                        dep_p2_ids[j * model.batch_size:(j + 1) * model.batch_size]]
            y_dict = rel_ids[j * model.batch_size:(j + 1) * model.batch_size]

            feed_dict = {
                model.path_length: path_dict,
                model.word_ids: word_dict,
                model.pos_ids: pos_dict,
                model.gr_ids: dep_dict,
                model.y: y_dict}
            _, loss, step = sess.run([model.opt, model.total_loss, model.global_step], feed_dict)
            if step % 10 == 0:
                print("Step:", step, "loss:", loss)
            '''
            if step % 1000 == 0:
                saver.save(sess, model_dir + '/model')
                print("Saved Model")
            '''
    all_predictions = []
    for j in range(num_batches):
        path_dict = [path1_len[j*model.batch_size:(j+1)*model.batch_size], path2_len[j*model.batch_size:(j+1)*model.batch_size]]
        word_dict = [word_p1_ids[j*model.batch_size:(j+1)*model.batch_size], word_p2_ids[j*model.batch_size:(j+1)*model.batch_size]]
        pos_dict = [pos_p1_ids[j*model.batch_size:(j+1)*model.batch_size], pos_p2_ids[j*model.batch_size:(j+1)*model.batch_size]]
        dep_dict = [dep_p1_ids[j*model.batch_size:(j+1)*model.batch_size], dep_p2_ids[j*model.batch_size:(j+1)*model.batch_size]]
        y_dict = rel_ids[j*model.batch_size:(j+1)*model.batch_size]

        feed_dict = {
            model.path_length:path_dict,
            model.word_ids:word_dict,
            model.pos_ids:pos_dict,
            model.gr_ids:dep_dict,
            model.y:y_dict}
        batch_predictions = sess.run(model.predictions, feed_dict)
        all_predictions.append(batch_predictions)

    y_pred = []
    for i in range(num_batches):
        for pred in all_predictions[i]:
            y_pred.append(pred)

    count = 0
    for i in range(model.batch_size*num_batches):
        count += y_pred[i]==rel_ids[i]
    accuracy = count/(model.batch_size*num_batches) * 100
    print("training accuracy", accuracy)