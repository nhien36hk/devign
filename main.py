# -*- coding: utf-8 -*-
"""
    This module is intended to join all the pipeline in separated tasks
    to be executed individually or in a flow by using command-line options

    Example:
    Dataset embedding and processing:
        $ python taskflows.py -e -pS
"""

import argparse
import gc
import os
import shutil
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

import configs
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def select(dataset):
    result = dataset.loc[dataset['project'] == "FFmpeg"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    #print(len(result))
    #result = result.iloc[11001:]
    #print(len(result))
    result = result.head(1000)

    return result

def create_task():
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    print("Total Functions: ", len(filtered))
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # cpg_files = [f for f in os.listdir(PATHS.cpg) if f.endswith('.bin')]
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = prepare.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    # json_files = [f for f in os.listdir(PATHS.cpg) if f.endswith('.json')]
    # Sort JSON files by their numeric prefix to match with slice indices
    json_files.sort(key=lambda x: int(x.split('_')[0]))
    print(json_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = prepare.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()


def embed_task():
    context = configs.Embed()
    # Tokenize source code into tokens
    dataset_files = data.get_directory_files(PATHS.cpg)
    w2vmodel = Word2Vec(**context.w2v_args)
    w2v_init = True
    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
        # Convert tokens to list for Word2Vec (Gensim 4.x compatibility)
        tokens_list = tokens_dataset.tokens.tolist()
        # word2vec used to learn the initial embedding of each token
        w2vmodel.build_vocab(corpus_iterable=tokens_list, update=not w2v_init)
        w2vmodel.train(corpus_iterable=tokens_list, total_examples=w2vmodel.corpus_count, epochs=1)
        if w2v_init:
            w2v_init = False
        # Embed cpg to node representation and pass to graph data structure
        cpg_dataset["nodes"] = cpg_dataset.apply(lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
        # remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        cpg_dataset["input"] = cpg_dataset.apply(lambda row: prepare.nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                                                                    w2vmodel.wv, context.edge_type), axis=1)
        data.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        data.write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
        del cpg_dataset
        gc.collect()
    print("Saving w2vmodel.")
    w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")


def process_task(stopping):
    context = configs.Process()
    devign = configs.Devign()
    model_path = PATHS.model + FILES.model
    model = process.Devign(path=model_path, device=DEVICE, model=devign.model, learning_rate=devign.learning_rate,
                           weight_decay=devign.weight_decay,
                           loss_lambda=devign.loss_lambda)
    train = process.Train(model, context.epochs)
    input_dataset = data.loads(PATHS.input)
    # split the dataset and pass to DataLoader with batch size
    train_loader, val_loader, test_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            data.train_val_test_split(input_dataset, shuffle=context.shuffle)))
    train_loader_step = process.LoaderStep("Train", train_loader, DEVICE)
    val_loader_step = process.LoaderStep("Validation", val_loader, DEVICE)
    test_loader_step = process.LoaderStep("Test", test_loader, DEVICE)

    if stopping:
        early_stopping = process.EarlyStopping(model, patience=context.patience)
        train(train_loader_step, val_loader_step, early_stopping)
        model.load()
    else:
        train(train_loader_step, val_loader_step)
        model.save()

    process.predict(model, test_loader_step)


def crossval_task(stopping, k_folds=10):
    context = configs.Process()
    devign = configs.Devign()
    input_dataset = data.loads(PATHS.input)

    folds = data.kfold_split(input_dataset, k=k_folds, shuffle=context.shuffle)
    results = []

    for fold_idx in range(k_folds):
        print(f"\n===== Fold {fold_idx + 1}/{k_folds} =====")

        test_df = folds[fold_idx]
        train_val_df = pd.concat([f for i, f in enumerate(folds) if i != fold_idx], ignore_index=True)

        train_set, val_set = data.train_val_split(train_val_df, train_ratio=0.8889, shuffle=context.shuffle)
        test_set = data.InputDataset(test_df)

        train_loader = train_set.get_loader(context.batch_size, shuffle=context.shuffle)
        val_loader = val_set.get_loader(context.batch_size, shuffle=context.shuffle)
        test_loader = test_set.get_loader(context.batch_size, shuffle=False)

        train_loader_step = process.LoaderStep("Train", train_loader, DEVICE)
        val_loader_step = process.LoaderStep("Validation", val_loader, DEVICE)
        test_loader_step = process.LoaderStep("Test", test_loader, DEVICE)

        model_path = PATHS.model + f"fold_{fold_idx}_" + FILES.model
        model = process.Devign(path=model_path, device=DEVICE, model=devign.model,
                               learning_rate=devign.learning_rate, weight_decay=devign.weight_decay,
                               loss_lambda=devign.loss_lambda)
        train = process.Train(model, context.epochs)

        if stopping:
            early_stopping = process.EarlyStopping(model, patience=context.patience)
            train(train_loader_step, val_loader_step, early_stopping)
            model.load()
        else:
            train(train_loader_step, val_loader_step)
            model.save()

        fold_metrics = process.predict(model, test_loader_step)
        results.append(fold_metrics)

        # Print per-fold summary
        print(
            "Fold metrics: "
            f"Acc {fold_metrics['Accuracy']:.4f}, "
            f"Pre {fold_metrics['Precision']:.4f}, "
            f"Rec {fold_metrics['Recall']:.4f}, "
            f"F1 {fold_metrics['F-measure']:.4f}"
        )

    if results:
        keys = ["Accuracy", "Precision", "Recall", "F-measure"]
        means = {k: float(np.mean([r[k] for r in results])) for k in keys}
        stds = {k: float(np.std([r[k] for r in results])) for k in keys}

        print("\nCross-validation summary (mean ± std):")
        print(
            f"Acc {means['Accuracy']:.4f} ± {stds['Accuracy']:.4f} | "
            f"Pre {means['Precision']:.4f} ± {stds['Precision']:.4f} | "
            f"Rec {means['Recall']:.4f} ± {stds['Recall']:.4f} | "
            f"F1 {means['F-measure']:.4f} ± {stds['F-measure']:.4f}"
        )

        print("Per-fold metrics:")
        print("Acc:", [round(r["Accuracy"], 4) for r in results])
        print("Pre:", [round(r["Precision"], 4) for r in results])
        print("Rec:", [round(r["Recall"], 4) for r in results])
        print("F1:", [round(r["F-measure"], 4) for r in results])


def main():
    """
    main function that executes tasks based on command-line options
    """
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument('-c', '--create', action='store_true')
    parser.add_argument('-e', '--embed', action='store_true')
    parser.add_argument('-p', '--process', action='store_true')
    parser.add_argument('-pS', '--process_stopping', action='store_true')
    parser.add_argument('-cv', '--crossval', action='store_true')
    parser.add_argument('-k', '--k_folds', type=int, default=10)

    args = parser.parse_args()

    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(False)
    if args.process_stopping:
        process_task(True)
    if args.crossval:
        crossval_task(stopping=args.process_stopping, k_folds=args.k_folds)



if __name__ == "__main__":
    main()
