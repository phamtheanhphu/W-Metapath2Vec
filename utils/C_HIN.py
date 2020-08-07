import json
import time
import os
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial import distance


class C_HIN():
    # node's attributes
    NODE_TITLE_ATTR = 'title'
    NODE_NAME_ATTR = 'name'
    NODE_TYPE_ATTR = 'type'
    NODE_IS_TEXT_BASED_NODE = 'is_text_based_node'
    NODE_TOPIC_DIST_VECTOR_ATTR = 'topic_dist_vector'

    # edge's attributes
    EDGE_REL_TYPE_ATTRIBUTE = 'type'

    # global configurations
    METAPATH_NODE_SEPARATOR = '-'
    ALLOW_SAME_SRC_DST_NODE_PATH_INSTANCE = True
    USE_STORED_PATH_INSTANCE_DATA = True
    '''
    define the limit number of node in each hop of metapath - fixing problem of memory limitation
    -> value = -1 -> no limitation
    '''
    LIMIT_NODESET_SIZE = 50

    def __init__(self, input_graph_file):

        self.input_graph_file = input_graph_file
        self.network = nx.read_gexf(self.input_graph_file)

        print('Reading given C-HIN from the given graph file: [{}]...'.format(self.input_graph_file))
        start_time = time.time()
        # deserialize topic distributions of each text-based node from json string -> list
        for (node_id, node_attrs) in self.network.nodes(data=True):
            if node_attrs[self.NODE_IS_TEXT_BASED_NODE] is True:
                if self.NODE_TOPIC_DIST_VECTOR_ATTR in node_attrs:
                    self.network.nodes[node_id][self.NODE_TOPIC_DIST_VECTOR_ATTR] = json.loads(
                        node_attrs[self.NODE_TOPIC_DIST_VECTOR_ATTR])

        self.nodes = self.network.nodes(data=True)
        self.edges = self.network.edges(data=True)
        print('-> Done [in: {:.3f} (seconds)], reading total: [{}] nodes and [{}] edges'
              .format((time.time() - start_time), len(self.nodes), len(self.edges)))

        self.parent_input_graph_file_dir = os.path.dirname(self.input_graph_file)

        self.output_path_instances_dir = os.path.join(self.parent_input_graph_file_dir, 'path_instances')
        if not os.path.exists(self.output_path_instances_dir):
            os.mkdir(self.output_path_instances_dir)
        self.input_graph_filename = Path(self.input_graph_file).stem

    def analyze_with_metapath(self, metapath):
        '''
        :param metapath: define meta-path for current C-HIN
        e.g. Author-Paper-Venue-Paper-Author, Author-Paper-Author, Venue-Paper-Author-Paper-Venue, etc.
        :return: extracted path instances from current C_HIN for the given meta-path
        '''
        self.metapath = metapath
        self.metapath_steps = self.metapath.split(self.METAPATH_NODE_SEPARATOR)
        self.output_path_instances_filepath = os.path.join(self.output_path_instances_dir, '{}.{}.json'
                                                           .format(self.input_graph_filename, self.metapath))
        start_time = time.time()

        if os.path.exists(self.output_path_instances_filepath) and self.USE_STORED_PATH_INSTANCE_DATA is True:
            print(
                'Existing path instances data file for metapath: [{}], loading data from file...'.format(self.metapath))
            with open(self.output_path_instances_filepath, 'r', encoding='utf-8') as f:
                self.path_instances = json.load(f)
            print(
                '-> Done [in: {:.3f} (seconds)], finish to load all path instances for metapath: [{}] from file, total: [{}]'
                    .format((time.time() - start_time), self.metapath, len(self.path_instances)))
        else:
            print('Analyzing given C-HIN with metapath: [{}]...'.format(self.metapath))
            if len(self.metapath_steps) == 0:
                raise Exception('Error', 'Invalid metapath')
            self.path_instances = []
            for (node_id, node_attrs) in self.nodes:
                self.__extract_path_instances(node_id)
            with open(self.output_path_instances_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.path_instances, f)
            print('-> Done [in: {:.3f} (seconds)], finish to extract all path instances for metapath: [{}], total: [{}]'
                  .format((time.time() - start_time), self.metapath, len(self.path_instances)))

    def __extract_path_instances(self, src_node_id):
        travelled_nodesets = {}
        cur_node_type_idx = 0
        if self.metapath_steps[cur_node_type_idx] != self.nodes[src_node_id][self.NODE_TYPE_ATTR]:
            return
        else:
            self.__do_metapath_traversal(src_node_id, cur_node_type_idx, travelled_nodesets)
            possible_path_instances = self.__generate_possible_path_instances(src_node_id, travelled_nodesets)

            for path_instance in possible_path_instances:
                if self.__is_valid_path_instance(path_instance):
                    self.path_instances.append(path_instance)
        return

    def __do_metapath_traversal(self, cur_node_id, cur_node_type_idx, travelled_nodesets):
        if cur_node_type_idx + 1 == len(self.metapath_steps):
            return
        next_type_nodeset = self.__find_next_type_nodeset_in_metapath(self.metapath_steps[cur_node_type_idx + 1],
                                                                      cur_node_id)
        if cur_node_type_idx not in travelled_nodesets.keys():
            travelled_nodesets[cur_node_type_idx] = next_type_nodeset
        else:
            travelled_nodesets[cur_node_type_idx] = list(
                set(travelled_nodesets[cur_node_type_idx]).union(next_type_nodeset))
        for next_node_id in next_type_nodeset:
            self.__do_metapath_traversal(next_node_id, cur_node_type_idx + 1, travelled_nodesets)

    def __find_next_type_nodeset_in_metapath(self, next_node_type, cur_node_id):
        neighbors = self.network.neighbors(cur_node_id)
        next_type_nodeset = []
        for node_id in neighbors:
            if self.nodes[node_id][self.NODE_TYPE_ATTR] == next_node_type:
                next_type_nodeset.append(node_id)
        return next_type_nodeset

    def __is_valid_path_instance(self, path_instance):
        # Checking the length of path instance and given meta-path
        if len(path_instance) != len(self.metapath_steps):
            return False
        # Checking if first and last node is the same
        if self.ALLOW_SAME_SRC_DST_NODE_PATH_INSTANCE is False:
            if path_instance[0] == path_instance[-1]:
                return False
        # Checking if (i) -> (i+1) is connected
        for i in range(0, len(path_instance) - 1):
            if not self.network.has_edge(path_instance[i], path_instance[i + 1]) and not self.network.has_edge(
                    path_instance[i + 1],
                    path_instance[i]):
                return False
        return True

    def __generate_possible_path_instances(self, src_node_id, travelled_nodesets):
        possible_path_instances = [[src_node_id]]
        for idx in travelled_nodesets.keys():
            next_type_nodesets = travelled_nodesets[idx]
            if self.LIMIT_NODESET_SIZE == -1:
                possible_path_instances = self.__concat_nodesets(
                    possible_path_instances, next_type_nodesets)
            else:
                possible_path_instances = self.__concat_nodesets(
                    possible_path_instances, next_type_nodesets[:self.LIMIT_NODESET_SIZE])
        return possible_path_instances

    def __concat_nodesets(self, previous, next):
        possible_path_instances = []
        for i in next:
            for j in previous:
                if type(j) is not list:
                    j = [j]
                possible_path_instances.append(j + [i])
        return possible_path_instances

    def is_content_based_metapath(self):
        existing_text_based_nodes = []
        for step in self.metapath_steps:
            for (node_id, node_attrs) in self.nodes:
                if node_attrs[self.NODE_TYPE_ATTR] == step and node_attrs[self.NODE_IS_TEXT_BASED_NODE] is True:
                    existing_text_based_nodes.append(step)
                    break
        if len(existing_text_based_nodes) >= 2 and len(existing_text_based_nodes) % 2 == 0:
            return True
        return False

    def calc_path_instance_weight(self, path_instance):

        sim_score = 0
        existing_text_based_nodes = []
        for node_id in path_instance:
            if self.nodes[node_id][self.NODE_IS_TEXT_BASED_NODE] is True:
                existing_text_based_nodes.append(node_id)

        if len(existing_text_based_nodes) >= 2 and len(existing_text_based_nodes) % 2 == 0:
            symmetric_sim_scores = []
            for idx, cur_node_id in enumerate(path_instance):
                if self.nodes[cur_node_id][self.NODE_IS_TEXT_BASED_NODE] is True and self.NODE_TOPIC_DIST_VECTOR_ATTR in \
                        self.nodes[cur_node_id]:
                    cur_symmetric_node_id = path_instance[-(idx + 1)]
                    if self.nodes[cur_symmetric_node_id][
                        self.NODE_IS_TEXT_BASED_NODE] is True and self.NODE_TOPIC_DIST_VECTOR_ATTR in self.nodes[
                        cur_symmetric_node_id]:
                        symmetric_sim_score = distance.cosine(self.nodes[cur_node_id][self.NODE_TOPIC_DIST_VECTOR_ATTR],
                                                              self.nodes[cur_symmetric_node_id][
                                                                  self.NODE_TOPIC_DIST_VECTOR_ATTR])
                        symmetric_sim_scores.append(symmetric_sim_score)
            if len(symmetric_sim_scores) > 0:
                sim_score = np.mean(symmetric_sim_scores)

        return sim_score
