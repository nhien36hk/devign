import sys
import re
from tree_sitter import Parser
from tree_sitter_languages import get_parser

# Load parsers
PARSER_C = get_parser("c")
PARSER_CPP = get_parser("cpp")

RE_NONALNUM = re.compile(r'[^0-9A-Za-z]+')
RE_CAMEL = re.compile(r'(?<!^)(?=[A-Z])')


def split_identifier(name: str):
    parts = [p for p in RE_NONALNUM.split(name) if p]
    subtokens = []
    for p in parts:
        chunks = RE_CAMEL.split(p)
        if len(chunks) > 1 and all(len(c) == 1 for c in chunks[:-1]):
            chunks = ["".join(chunks[:-1]), chunks[-1]]
        subtokens.extend([c.lower() for c in chunks if c])
    return subtokens


def parser_for_path(path: str):
    path = path.lower()
    if path.endswith((".c", ".h")):
        return PARSER_C
    if path.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hh")):
        return PARSER_CPP
    return None


def is_function_node(node):
    return node.type == "function_definition" or node.type == "function_declaration"


def normalize_leaf(node, code):
    t = node.type
    raw = code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()
    if not raw:
        return []
    
    # identifier -> split subtokens
    if t == "identifier":
        return split_identifier(raw)
    # string/char literal -> <STR>
    if "string" in t or "char" in t:
        return ["<STR>"]
    # number literal -> <NUM>
    if "number" in t or "integer" in t or "float" in t:
        return ["<NUM>"]
    # Skip quotes and escape sequences - they're part of string literals
    if t == '"' or t == "'" or t == "escape_sequence":
        return []
    # otherwise keep the raw token (keywords, operators, punctuation)
    return [raw]


def tokens_from_node(code, node):
    tokens = []
    stack = [node]
    in_string = False
    
    while stack:
        n = stack.pop()
        if n.type == "comment":
            continue

        # Handle string literals
        is_string = n.type == '"' or n.type == "'" or n.type == "char_literal"
        if is_string and not in_string:
            in_string = True
            tokens.append("<STR>")
            continue
        elif is_string and in_string:
            in_string = False
            continue
        elif in_string:
            continue
        
        if n.child_count == 0:
            tokens.extend(normalize_leaf(n, code))
        else:
            for i in range(n.child_count - 1, -1, -1):
                stack.append(n.children[i])
    return tokens


def extract_functions(code_bytes, parser):
    tree = parser.parse(code_bytes)
    root = tree.root_node
    stack = [root]
    funcs = []
    
    while stack:
        n = stack.pop()
        if is_function_node(n):
            funcs.append(n)
        for i in range(n.child_count - 1, -1, -1):
            stack.append(n.children[i])
    return funcs
