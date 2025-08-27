import json
import re
import subprocess
import os.path
import os
import time
from .cpg_client_wrapper import CPGClientWrapper
#from ..data import datamanager as data


def funcs_to_graphs(funcs_path):
    client = CPGClientWrapper()
    # query the cpg for the dataset
    print(f"Creating CPG.")
    graphs_string = client(funcs_path)
    # removes unnecessary namespace for object references
    graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
    graphs_json = json.loads(graphs_string)

    return graphs_json["functions"]


def graph_indexing(graph):
    idx = int(graph["file"].split(".c")[0].split("/")[-1])
    del graph["file"]
    return idx, {"functions": [graph]}


def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    env = os.environ.copy()
    env["JAVA_TOOL_OPTIONS"] = "-Xmx16g -Xms2g"
    joern_parse_call = subprocess.run(["./" + joern_path + "joern-parse", input_path, "--out", output_path + out_file],
                                      stdout=subprocess.PIPE, text=True, check=True, env=env)
    print(str(joern_parse_call))
    return out_file


def joern_create(joern_path, in_path, out_path, cpg_files):
    json_files = []
    script_path = f"{os.path.dirname(os.path.abspath(joern_path))}/graph-for-funcs.sc"

    for cpg_file in cpg_files:
        json_file_name = f"{cpg_file.split('.')[0]}.json"
        json_out = f"{os.path.abspath(out_path)}/{json_file_name}"
        cpg_abs = f"{os.path.abspath(in_path)}/{cpg_file}"

        commands = [
            "./" + joern_path + "joern",
            "--script", script_path,
            "--params", f"cpgFile={cpg_abs},outputFile={json_out}"
        ]

        env = os.environ.copy()
        env["JAVA_TOOL_OPTIONS"] = "-Xmx16g -Xms2g"

        try:
            proc = subprocess.run(
                commands,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,
                check=True,
                env=env
            )
            if proc.stderr and "WARN" not in proc.stderr:
                print(proc.stderr)

            json_files.append(json_file_name)
            print(f"✅ {json_file_name}")
        except subprocess.TimeoutExpired:
            print(f"❌ timeout: {cpg_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ failed: {cpg_file}\n{e.stderr}")
    return json_files


def json_process(in_path, json_file):
    if os.path.exists(in_path+json_file):
        with open(in_path+json_file) as jf:
            cpg_string = jf.read()
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
            cpg_json = json.loads(cpg_string)
            container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
            return container
    return None

'''
def generate(dataset, funcs_path):
    dataset_size = len(dataset)
    print("Size: ", dataset_size)
    graphs = funcs_to_graphs(funcs_path[2:])
    print(f"Processing CPG.")
    container = [graph_indexing(graph) for graph in graphs["functions"] if graph["file"] != "N/A"]
    graph_dataset = data.create_with_index(container, ["Index", "cpg"])
    print(f"Dataset processed.")

    return data.inner_join_by_index(dataset, graph_dataset)
'''

# client = CPGClientWrapper()
# client.create_cpg("../../data/joern/")
# joern_parse("../../joern/joern-cli/", "../../data/joern/", "../../joern/joern-cli/", "gen_test")
# print(funcs_to_graphs("/data/joern/"))
"""
while True:
    raw = input("query: ")
    response = client.query(raw)
    print(response)
"""
