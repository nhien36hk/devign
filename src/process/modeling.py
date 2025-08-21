import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from ..utils import log as logger
from ..utils.objects.metrics import Metrics


class Train(object):
    def __init__(self, step, epochs, verbose=True):
        self.epochs = epochs
        self.step = step
        self.history = History()
        self.verbose = verbose

    def __call__(self, train_loader_step, val_loader_step=None, early_stopping=None):
        for epoch in range(self.epochs):
            self.step.train()
            train_stats = train_loader_step(self.step)
            self.history(train_stats, epoch + 1)

            if val_loader_step is not None:
                with torch.no_grad():
                    self.step.eval()
                    val_stats = val_loader_step(self.step)
                    self.history(val_stats, epoch + 1)

                print(self.history)

                # Compute epoch metrics for Validation only
                val_metrics = Metrics(val_stats.outs(), val_stats.labels())
                v = val_metrics()
                logger.log_info(
                    'metrics-epoch',
                    f"Val (Acc {v['Accuracy']:.3f}, Pre {v['Precision']:.3f}, Rec {v['Recall']:.3f}, F1 {v['F-measure']:.3f})"
                )

                if early_stopping is not None:
                    valid_loss = val_stats.loss()
                    # early_stopping needs the validation loss to check if it has decreased,
                    # and if it has, it will make a checkpoint of the current model
                    if early_stopping(valid_loss):
                        self.history.log()
                        self.history.plot_losses()
                        self.history.plot_validation_metrics()
                        return
            else:
                print(self.history)
        self.history.log()
        self.history.plot_losses()
        self.history.plot_validation_metrics()


def predict(step, test_loader_step):
    print(f"Testing")
    with torch.no_grad():
        step.eval()
        stats = test_loader_step(step)
        metrics = Metrics(stats.outs(), stats.labels())
        print(metrics)
        metrics.log()
            
        # Plot test metrics
        history = History()  # Create temporary history object to access plot method
        history.plot_test_metrics(stats)
        
    return metrics()


class History:
    def __init__(self):
        self.history = {}
        self.epoch = 0
        self.timer = time.time()
        self.val_metrics = []
        self.test_metrics = []

    def __call__(self, stats, epoch):
        self.epoch = epoch

        if epoch in self.history:
            self.history[epoch].append(stats)
        else:
            self.history[epoch] = [stats]

        # Capture validation metrics when Validation entry is available for the epoch
        current_stats = self.history[epoch]
        if len(current_stats) >= 1:
            if getattr(current_stats[-1], 'name', None) == 'Validation':
                # Compute metrics for validation only
                val_stat = current_stats[-1]
                v = Metrics(val_stat.outs(), val_stat.labels())()
                self.val_metrics.append(v)

    def __str__(self):
        epoch = f"\nEpoch {self.epoch};"
        stats = ' - '.join([f"{res}" for res in self.current()])
        timer = f"Time: {(time.time() - self.timer)}"

        return f"{epoch} - {stats} - {timer}"

    def current(self):
        return self.history[self.epoch]

    def log(self):
        msg = f"(Epoch: {self.epoch}) {' - '.join([f'({res})' for res in self.current()])}"
        logger.log_info("history", msg)

    def epoch_losses(self):
        train_losses = []
        val_losses = []
        for epoch in sorted(self.history.keys()):
            stats_list = self.history[epoch]
            train_loss = None
            val_loss = None
            for st in stats_list:
                if getattr(st, 'name', None) == 'Train':
                    train_loss = st.loss()
                elif getattr(st, 'name', None) == 'Validation':
                    val_loss = st.loss()
            if train_loss is not None:
                train_losses.append(train_loss)
            if val_loss is not None:
                val_losses.append(val_loss)
        return train_losses, val_losses

    def plot_losses(self, save_dir: str = 'workspace', filename: str = 'loss_curve.png'):
        train_losses, val_losses = self.epoch_losses()
        if not train_losses and not val_losses:
            return
        os.makedirs(save_dir, exist_ok=True)
        epochs_train = list(range(1, len(train_losses) + 1))
        plt.figure(figsize=(8, 5))
        if train_losses:
            plt.plot(epochs_train, train_losses, label='Train Loss', marker='o', linewidth=1.5)
        if val_losses:
            epochs_val = list(range(1, len(val_losses) + 1))
            plt.plot(epochs_val, val_losses, label='Validation Loss', marker='o', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved loss curve to {save_path}")

    def plot_validation_metrics(self, save_dir: str = 'workspace'):
        if not self.val_metrics:
            return
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract series for metrics
        def series(metrics_list, key):
            return [m.get(key, None) for m in metrics_list]
        epochs = list(range(1, len(self.val_metrics) + 1))

        # Plot all validation metrics in one figure
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, series(self.val_metrics, 'Accuracy'), label='Validation Accuracy', marker='o', linewidth=2)
        plt.plot(epochs, series(self.val_metrics, 'Precision'), label='Validation Precision', marker='x', linewidth=2)
        plt.plot(epochs, series(self.val_metrics, 'Recall'), label='Validation Recall', marker='s', linewidth=2)
        plt.plot(epochs, series(self.val_metrics, 'F-measure'), label='Validation F1-Score', marker='^', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics per Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        save_path = os.path.join(save_dir, 'validation_metrics.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved validation metrics curve to {save_path}")

    def plot_test_metrics(self, test_stats, save_dir: str = 'workspace'):
        """Plot test metrics after training is complete"""
        if not test_stats:
            return
        os.makedirs(save_dir, exist_ok=True)
        
        # Compute test metrics
        test_metrics = Metrics(test_stats.outs(), test_stats.labels())
        test_m = test_metrics()
        
        # Create bar chart for test metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [test_m['Accuracy'], test_m['Precision'], test_m['Recall'], test_m['F-measure']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Test Set Performance')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.4)
        save_path = os.path.join(save_dir, 'test_metrics.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved test metrics to {save_path}")
        
        # Also print test metrics summary
        print(f"\n📊 TEST SET RESULTS:")
        print(f"   Accuracy: {test_m['Accuracy']:.4f}")
        print(f"   Precision: {test_m['Precision']:.4f}")
        print(f"   Recall: {test_m['Recall']:.4f}")
        print(f"   F1-Score: {test_m['F-measure']:.4f}")
        print(f"   AUC: {test_m['AUC']:.4f}")
        
        # Create comprehensive results table
        self.create_final_results_table(test_m, save_dir)

    def create_final_results_table(self, test_metrics, save_dir: str = 'workspace'):
        """Create comprehensive results table for paper comparison"""
        
        # Get best validation metrics (for reference only, not in table)
        if self.val_metrics:
            best_val_epoch = np.argmax([m['F-measure'] for m in self.val_metrics])
            best_val_metrics = self.val_metrics[best_val_epoch]
        else:
            best_val_metrics = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F-measure': 0, 'AUC': 0}
        
        # Create results table
        results_data = {
            'Metric': [
                'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'
            ],
            'Test Set': [
                f"{test_metrics['Accuracy']:.4f}",
                f"{test_metrics['Precision']:.4f}",
                f"{test_metrics['Recall']:.4f}",
                f"{test_metrics['F-measure']:.4f}",
                f"{test_metrics['AUC']:.4f}"
            ]
        }
        
        df = pd.DataFrame(results_data)
        
        # Save as CSV
        csv_path = os.path.join(save_dir, 'final_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved final results table to {csv_path}")
        
        # Create LaTeX table for paper
        latex_table = df.to_latex(index=False, float_format="%.4f")
        latex_path = os.path.join(save_dir, 'final_results.tex')
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Comparison on Vulnerability Detection}\n")
            f.write("\\label{tab:results}\n")
            f.write(latex_table)
            f.write("\\end{table}\n")
        print(f"Saved LaTeX table to {latex_path}")
        
       