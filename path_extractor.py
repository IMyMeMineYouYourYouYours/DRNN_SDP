from nltk.parse.stanford import StanfordDependencyParser
import _pickle

dep_parser = StanfordDependencyParser(path_to_jar="C:/Users/Ailab_cho/PycharmProjects/DRNN_SDP/stanford-parser-full-2018-10-17/stanford-parser-full-2018-10-17/stanford-parser.jar", path_to_models_jar="C:/Users/Ailab_cho/PycharmProjects/DRNN_SDP/stanford-parser-full-2018-10-17/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar")

def lca(tree, index1, index2):
    node = index1
    path1 = []
    path2 = []
    path1.append(index1)
    path2.append(index2)
    while(node!=tree.root):
        node = tree.nodes[node['head']]
        path1.append(node)
    node = index2
    while(node!=tree.root):
        node = tree.nodes[node['head']]
        path2.append(node)
    for l1, l2 in zip(path1[::-1], path2[::-1]):
        if(l1==l2):
            temp=l1
    return temp

def path_lca(tree, node, lca_node):
    path = []
    path.append(node)
    while(node!=lca_node):
        node = tree.nodes[node['head']]
        path.append(node)
    return path
def load_path(path, data):
    f = open(path + "/" + data + "_data", 'rb')
    sentences, e1, e2 = _pickle.load(f)
    f.close()
    '''
    print(sentences[7588])
    print(sentences[2608])
    '''
    word_path1 = [0]*8000
    word_path2 = [0]*8000
    rel_path1 = [0]*8000
    rel_path2 = [0]*8000
    pos_path1 = [0]*8000
    pos_path2 = [0]*8000

    for i in range(8000):
        try:
            parse_tree = dep_parser.raw_parse(sentences[i])
            for trees in parse_tree:
                tree = trees
            node1 = tree.nodes[e1[i] + 1]
            node2 = tree.nodes[e2[i] + 1]
            if node1['address'] != None and node2['address'] != None:
                print(i, "success")
                lca_node = lca(tree, node1, node2)
                path1 = path_lca(tree, node1, lca_node)
                path2 = path_lca(tree, node2, lca_node)

                word_path1[i] = [p["word"] for p in path1]
                word_path2[i] = [p["word"] for p in path2]
                rel_path1[i] = [p["rel"] for p in path1]
                rel_path2[i] = [p["rel"] for p in path2]
                pos_path1[i] = [p["tag"] for p in path1]
                pos_path2[i] = [p["tag"] for p in path2]
            else:
                print(i, node1["address"], node2["address"])
        except AssertionError:
            print(i, "error")

    file = open(path + "/" + data + "_paths", 'wb')
    _pickle.dump([word_path1, word_path2, rel_path1, rel_path2, pos_path1, pos_path2], file)

load_path('SemEval2010_task8_all_data/SemEval2010_task8_all_data/SemEval2010_task8_training', "TRAIN")