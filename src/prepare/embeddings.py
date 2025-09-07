import numpy as np
import torch
from torch_geometric.data import Data
from src.utils.objects.cpg.node import get_type
from gensim.models.keyedvectors import Word2VecKeyedVectors
from src.utils.functions.token import tokens_from_node, parser_for_path
from tree_sitter_languages import get_parser


class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.nodes_dim = nodes_dim
        self.mapping = {} 
        self.parser = get_parser("c")

        assert self.nodes_dim >= 0

    def __call__(self, nodes):
        embedded_nodes = self.embed_nodes(nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        return nodes_tensor


    def embed_nodes(self, nodes):
        embeddings = []

        for i, node in enumerate(nodes):
            node_id = node["id"]
            self.mapping[node_id] = i
            
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
    def __init__(self, id_mapping):
        self.id_mapping = id_mapping

    def __call__(self, edges):
        return torch.tensor(self.build_connections(edges)).long()

    def build_connections(self, edges):
        coo = [[], []]
        
        for edge in edges:
            src_id, tgt_id = edge[0], edge[1]
            
            if src_id in self.id_mapping and tgt_id in self.id_mapping:
                src_idx = self.id_mapping[src_id]
                tgt_idx = self.id_mapping[tgt_id]
                coo[0].append(src_idx)
                coo[1].append(tgt_idx)
            else:
                print(f"Edge {src_id} or {tgt_id} not in id_mapping")
                
        return coo


def nodes_to_input(cpg, target, nodes_dim, keyed_vectors):

    node_embed = NodesEmbedding(nodes_dim, keyed_vectors)
    x = node_embed(cpg["nodes"])
    
    graph_embed = GraphsEmbedding(node_embed.mapping)
    label = torch.tensor([target]).float()
    
    ast_edges = graph_embed(cpg["ast_edges"])
    cfg_edges = graph_embed(cpg["cfg_edges"])
    cdg_edges = graph_embed(cpg["cdg_edges"])
    ddg_edges = graph_embed(cpg["ddg_edges"])

    return Data(
        x=x,
        ast_edge_index=ast_edges,
        cfg_edge_index=cfg_edges,
        cdg_edge_index=cdg_edges,
        ddg_edge_index=ddg_edges,
        y=label)
