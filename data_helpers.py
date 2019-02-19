import re, sys, nltk
import numpy as np
import _pickle as pickle

def load_data(path, data):
    lines = []
    for line in open(path+"/"+data+"_FILE.txt", 'r'):
        lines.append(line.strip())

    relations = []
    for i, w in enumerate(lines):
        if((i+3)%4==0):
            relations.append(w)

    f = open(path+"/"+data+"_relation.txt", 'w')
    for rel in relations:
        f.write(rel+'\n')

    lines = []
    for line in open(path+"/"+data+"_FILE.txt", 'r'):
        m = re.match(r'^([0-9]+)\s"(.+)"$', line.strip())
        if(m is not None):
            lines.append(m.group(2))

    sentences = []
    e1 = []
    e2 = []
    for j, line in enumerate(lines):
        text = []
        temp = []
        t = line.split("<e1>")
        text.append(t[0])
        temp.append(t[0])

        t = t[1].split("</e1>")
        e1_text = text
        e1_text = " ".join(e1_text)
        e1_text = nltk.word_tokenize(e1_text)
        text.append(t[0])
        e11 = t[0]
        y = nltk.word_tokenize(t[0])
        y[0] += "E11"
        temp.append(" ".join(y))
        t = t[1].split("<e2>")
        text.append(t[0])
        temp.append(t[0])
        t = t[1].split("</e2>")
        e22 = t[0]
        e2_text = text
        e2_text = " ".join(e2_text)
        e2_text = nltk.word_tokenize(e2_text)
        text.append(t[0])
        text.append(t[1])
        y = nltk.word_tokenize(t[0])
        y[0] += "E22"
        temp.append(" ".join(y))
        temp.append(t[1])

        text = " ".join(text)
        text = nltk.word_tokenize(text)
        temp = " ".join(temp)
        temp = nltk.word_tokenize(temp)

        q1 = nltk.word_tokenize(e11)[0]
        q2 = nltk.word_tokenize(e22)[0]
        for i, word in enumerate(text):
            if (word.find(q1) != -1):
                if (temp[i].find("E11") != -1):
                    e1.append(i)
                    break
        for i, word in enumerate(text):
            if (word.find(q2) != -1):
                if (temp[i].find("E22") != -1):
                    e2.append(i)
        text = " ".join(text)
        sentences.append(text)
        print(j, text)

    len(sentences), len(e1), len(e2)

    with open(path+'/'+data+"_data", "wb") as f:
        pickle.dump((sentences, e1, e2), f)
        f.close()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


load_data('SemEval2010_task8_all_data/SemEval2010_task8_all_data/SemEval2010_task8_training', "TRAIN")
load_data('SemEval2010_task8_all_data/SemEval2010_task8_all_data/SemEval2010_task8_testing', "TEST")