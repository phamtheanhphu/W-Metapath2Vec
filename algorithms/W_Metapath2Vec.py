import numpy as np
import operator
import time
from utils.C_HIN import C_HIN
from gensim.models import Word2Vec


class W_Metapath2Vec():

    def __init__(self, input_graph_file, output_emb_file, metapath, walk_per_node=80, walk_length=100, window_size=5,
                 embedding_dimensions=128, epoch=1):

        # Reading the content-based heterogeneous information network (C-HIN)
        self.input_graph_file = input_graph_file
        self.output_emb_file = output_emb_file
        self.C_HIN = C_HIN(self.input_graph_file)

        self.metapath = metapath
        self.walk_per_node = walk_per_node
        self.walk_length = walk_length
        self.window_size = window_size
        self.embedding_dimensions = embedding_dimensions
        self.epoch = epoch

        self.unnormized_metapath_trans_probs = {}
        self.normized_metapath_trans_probs = {}
        self.lambda_constant = None

        self.walks = None

    def train(self):
        # Analyzing path instances with the given metapath and init transitional probabilities
        self.__preprocess_network()
        # Simulate the walks from the given C-HIN
        self.__simulate_walks()
        # Learn the representations from the given C-HIN
        self.__learn_node_embedding()

    def __preprocess_network(self):
        self.C_HIN.analyze_with_metapath(self.metapath)
        self.__initialize_trans_probs()

    def __initialize_trans_probs(self):
        '''
        :return: initializing the transitional probability for pairwise nodes x -> v
        the transitional probabilities is identified by the equation (3)
        '''
        for path_instance in self.C_HIN.path_instances:
            different_type_node_trans_probs = []
            for i in range(0, len(path_instance) - 1):
                for j in range(1, len(path_instance) - 1):
                    if self.C_HIN.network.has_edge(path_instance[i], path_instance[j]):
                        different_type_node_trans_prob = 1
                        if len(list(self.C_HIN.network.neighbors(path_instance[i]))) > 0:
                            different_type_node_trans_prob = 1 / len(
                                list(self.C_HIN.network.neighbors(path_instance[i])))
                        self.unnormized_metapath_trans_probs[
                            (path_instance[i], path_instance[j])] = different_type_node_trans_prob
                        different_type_node_trans_probs.append(different_type_node_trans_prob)
                sim_cs = self.C_HIN.calc_path_instance_weight(path_instance)
                same_type_node_trans_prob = sum(different_type_node_trans_probs) + sim_cs
                self.unnormized_metapath_trans_probs[(path_instance[0], path_instance[-1])] = same_type_node_trans_prob

        # calculating the global normalized constant (lambda)
        self.lambda_constant = sum([float(self.unnormized_metapath_trans_probs[i])
                                    for i in self.unnormized_metapath_trans_probs.keys()])

        for (x, v) in self.unnormized_metapath_trans_probs:
            self.normized_metapath_trans_probs[(x, v)] = self.unnormized_metapath_trans_probs[
                                                             (x, v)] / self.lambda_constant

    def __simulate_walks(self):
        self.walks = []
        for walk_iter in range(self.walk_per_node):
            for (node_id, node_attrs) in self.C_HIN.nodes:
                if node_attrs[self.C_HIN.NODE_TYPE_ATTR] == self.C_HIN.metapath_steps[0]:
                    walk = self.__w_metapath2vec_random_walk(node_id)
                    self.walks.append(walk)

    def __w_metapath2vec_random_walk(self, node_id):
        walk = [node_id]
        acc = 1
        next_node_type = self.C_HIN.metapath_steps[acc]
        is_restart = False
        while len(walk) < self.walk_length:
            cur_node_id = walk[-1]
            next_node = self.__identify_next_node(walk, cur_node_id, next_node_type, is_restart)
            if next_node is not None:
                if is_restart is True: is_restart = False
                walk.append(next_node)
            else:
                break
            if acc == len(self.C_HIN.metapath_steps) - 1 and len(walk) > 2:
                acc = 1
                is_restart = True
            else:
                acc += 1
            next_node_type = self.C_HIN.metapath_steps[acc]
        return walk

    def __identify_next_node(self, walk, cur_node_id, next_node_type, is_restart):
        trans_probs = {}
        next_node_id = None
        if is_restart == True:
            next_type_neighbors = []
            for neighbor in self.C_HIN.network.neighbors(cur_node_id):
                if self.C_HIN.nodes[neighbor][self.C_HIN.NODE_TYPE_ATTR] == next_node_type:
                    next_type_neighbors.append(neighbor)
            next_node_id = next_type_neighbors[np.random.randint(len(next_type_neighbors))]
        else:
            if next_node_type == self.C_HIN.metapath_steps[-1] and len(walk) >= (len(self.C_HIN.metapath_steps) - 1):
                for (x, v) in self.normized_metapath_trans_probs:
                    if x == walk[-(len(self.C_HIN.metapath_steps) - 1)] and self.C_HIN.nodes[v][
                        self.C_HIN.NODE_TYPE_ATTR] == next_node_type and v not in walk:
                        trans_probs[v] = (self.normized_metapath_trans_probs[(x, v)])
            else:
                for (x, v) in self.normized_metapath_trans_probs:
                    if x == cur_node_id and self.C_HIN.nodes[v][
                        self.C_HIN.NODE_TYPE_ATTR] == next_node_type and v not in walk:
                        trans_probs[v] = (self.normized_metapath_trans_probs[(x, v)])

            if len(trans_probs.keys()) > 0:
                next_node_id = max(trans_probs.items(), key=operator.itemgetter(1))[0]

        return next_node_id

    def __learn_node_embedding(self):
        '''
        Multiple representation learning techniques can be applied in this proccess
        for the heterogeneous Skip-gram architecture C/C++ code from Dong, Yuxiao et al. (Metapath2Vec, 2017)
        is available at this repository: https://www.dropbox.com/s/w3wmo2ru9kpk39n/code_metapath2vec.zip?dl=0
        In this version of W-Metapath2Vec source code, we simply applied the traditional Word2Vec model of GenSim library
        :return: node embeddings for the given C-HIN following the generated walks
        '''
        print('Start to learn the node embedding from the given C-HIN...')
        start_time = time.time()
        model = Word2Vec(self.walks,
                         size=self.embedding_dimensions,
                         window=self.window_size,
                         min_count=0,
                         sg=1,
                         workers=8,
                         iter=self.epoch)

        model.wv.save_word2vec_format(self.output_emb_file)
        print('-> Done [in: {:.3f} (seconds)], finish to learn node embedding and save to: [{}]'
              .format((time.time() - start_time), self.output_emb_file))
