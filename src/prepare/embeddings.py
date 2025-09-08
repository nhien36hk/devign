import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from src.utils.objects.cpg.node import get_type
from gensim.models.keyedvectors import Word2VecKeyedVectors
from src.utils.functions.token import tokens_from_node, parser_for_path
from tree_sitter_languages import get_parser


class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.nodes_dim = nodes_dim
        self.node_info = {}
        self.parser = get_parser("c")

        assert self.nodes_dim >= 0

    def __call__(self, node_type, nodes):
        embedded_nodes = self.embed_nodes(node_type, nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        return nodes_tensor


    def embed_nodes(self, node_type, nodes):
        embeddings = []

        for i, node in enumerate(nodes):
            self.node_info[node["id"]] = {"index": i, "node_type": node_type}
            
            # Get node's code
            node_code = node.get("code", "")
            if not node_code:
                embeddings.append(np.zeros(self.kv_size + 1))
                continue
                
            # Tokenize using tree-sitter
            tokens = self.tokenize_node_code(node_code)
            if not tokens:  
                embeddings.append(np.zeros(self.kv_size + 1))
                continue
                
            # Get each token's learned embedding vector
            vectors = np.array(self.get_vectors(tokens))
            # The node's source embedding is the average of it's embedded tokens
            src_embed = np.mean(vectors, 0)
            # The node representation is the concatenation of label and source embeddings
            node_type = get_type(node["label"])
            embed = np.concatenate((np.array([node_type]), src_embed), axis=0)
            embeddings.append(embed)

        return np.array(embeddings)

    def get_vectors(self, tokens):
        vectors = []

        for token in tokens:
            if token in self.w2v_keyed_vectors.key_to_index:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                vectors.append(np.zeros(self.kv_size))

        return vectors

    def tokenize_node_code(self, node_code):
        try:
            code_bytes = node_code.encode("utf-8", errors="ignore")
            tree = self.parser.parse(code_bytes)
            tokens = tokens_from_node(code_bytes, tree.root_node)
            return tokens
        except Exception as e:
            print(f"Error tokenizing node code '{node_code}': {e}")
            return []


class GraphsEmbedding:
    def __init__(self, node_info):
        self.node_info = node_info
        self.relations = {}

    def __call__(self, edge_type, edges):
        self.build_connections(edge_type, edges)

    def build_connections(self, edge_type, edges):
        for edge in edges:
            src_id, dst_id = edge[0], edge[1]
            
            if src_id not in self.node_info or dst_id not in self.node_info:
                print(f"Warning: Edge {src_id} -> {dst_id} references missing node(s)")
                continue
                
            src_info = self.node_info[src_id]
            dst_info = self.node_info[dst_id]

            relation_key = (src_info['node_type'], edge_type, dst_info['node_type'])

            if relation_key not in self.relations:
                self.relations[relation_key] = ([], [])
            
            self.relations[relation_key][0].append(src_info['index'])
            self.relations[relation_key][1].append(dst_info['index'])


def nodes_to_input(cpg, target, nodes_dim, keyed_vectors):
    data = HeteroData()

    node_embed = NodesEmbedding(nodes_dim, keyed_vectors)
    for node_type, nodes in cpg["nodes"].items():
        data[node_type].x = node_embed(node_type, nodes)
    
    graph_embed = GraphsEmbedding(node_embed.node_info)
    data.y = torch.tensor([target]).float()

    for edge_type, edges in cpg["edges"].items():   
        graph_embed(edge_type, edges)
    
    for relation_key, (src_indices, dst_indices) in graph_embed.relations.items():
        src_type, edge_type, dst_type = relation_key        
        data[src_type, edge_type, dst_type].edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    
    return data 
