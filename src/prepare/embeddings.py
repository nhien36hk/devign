import numpy as np
import torch
from torch_geometric.data import Data
from src.utils.functions.parse import tokenizer
from src.utils.objects.cpg.node import get_type
from gensim.models.keyedvectors import Word2VecKeyedVectors


class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.nodes_dim = nodes_dim
        self.mapping = {} 

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
            # Tokenize the code
            tokens = tokenizer(node_code, True)
            if not tokens:
                msg = f"Empty TOKENIZED from node CODE {node_code}"
                print(msg)
                continue
                
            # Get each token's learned embedding vector
            vectors = np.array(self.get_vectors(tokens, node))
            # The node's source embedding is the average of it's embedded tokens
            src_embed = np.mean(vectors, 0)
            # The node representation is the concatenation of label and source embeddings
            node_type = get_type(node["label"])
            embed = np.concatenate((np.array([node_type]), src_embed), axis=0)
            embeddings.append(embed)

        return np.array(embeddings)

    def get_vectors(self, tokens, node):
        vectors = []
        node_label = node["label"]
        node_code = node.get("code", "")

        for token in tokens:
            if token in self.w2v_keyed_vectors.key_to_index:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                vectors.append(np.zeros(self.kv_size))
                if node_label not in ["IDENTIFIER", "LITERAL", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT"]:
                    msg = f"No vector for TOKEN {token} in {node_code}."
                    print(msg)

        return vectors


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

    # Combine all edge types into single edge_index
    all_edges = []
    if "ast_edges" in cpg:
        all_edges.extend(cpg["ast_edges"])
    if "cfg_edges" in cpg:
        all_edges.extend(cpg["cfg_edges"])
    if "cdg_edges" in cpg:
        all_edges.extend(cpg["cdg_edges"])
    if "ddg_edges" in cpg:
        all_edges.extend(cpg["ddg_edges"])
    
    edge_index = graph_embed(all_edges)

    return Data(
        x=x,
        edge_index=edge_index,
        y=label)
