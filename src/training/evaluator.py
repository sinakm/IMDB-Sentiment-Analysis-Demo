"""
Model evaluation utilities for sentiment analysis.

This module provides comprehensive evaluation capabilities for all three
model approaches with detailed metrics and analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """
    Comprehensive model evaluator for sentiment analysis models.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: Optional[torch.device] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def _get_model_predictions(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions and probabilities.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            # Handle different model types
            if hasattr(self.model, 'create_verbalizer_input'):
                # Verbalizer model
                logits = self.model(**batch)
            else:
                # LSTM models
                logits, _ = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    lengths=batch.get('lengths')
                )
            
            if logits.dim() > 1:
                logits = logits.squeeze()
            
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()
            
            return predictions, probabilities
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                predictions, probabilities = self._get_model_predictions(batch)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(batch['labels'].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Ensure targets are integers
        if all_targets.dtype == np.float32 or all_targets.dtype == np.float64:
            all_targets = all_targets.astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions, average='binary'),
            'recall': recall_score(all_targets, all_predictions, average='binary'),
            'f1': f1_score(all_targets, all_predictions, average='binary'),
            'auc': roc_auc_score(all_targets, all_probabilities)
        }
        
        return metrics
    
    def detailed_evaluation(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Perform detailed evaluation with additional analysis.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with detailed evaluation results
        """
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_lengths = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Detailed Evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                predictions, probabilities = self._get_model_predictions(batch)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(batch['labels'].cpu().numpy())
                
                # Store lengths if available
                if 'lengths' in batch:
                    all_lengths.extend(batch['lengths'].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Ensure targets are integers
        if all_targets.dtype == np.float32 or all_targets.dtype == np.float64:
            all_targets = all_targets.astype(int)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions, average='binary'),
            'recall': recall_score(all_targets, all_predictions, average='binary'),
            'f1': f1_score(all_targets, all_predictions, average='binary'),
            'auc': roc_auc_score(all_targets, all_probabilities)
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_targets, all_probabilities)
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence(
            all_predictions, all_probabilities, all_targets
        )
        
        # Length analysis (if available)
        length_analysis = None
        if all_lengths:
            length_analysis = self._analyze_by_length(
                all_predictions, all_targets, np.array(all_lengths)
            )
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'confidence_analysis': confidence_analysis,
            'length_analysis': length_analysis,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets
        }
    
    def _analyze_confidence(self, 
                          predictions: np.ndarray, 
                          probabilities: np.ndarray, 
                          targets: np.ndarray) -> Dict[str, Any]:
        """
        Analyze prediction confidence.
        
        Args:
            predictions: Model predictions
            probabilities: Model probabilities
            targets: True targets
            
        Returns:
            Confidence analysis results
        """
        # Calculate confidence (distance from 0.5)
        confidence = np.abs(probabilities - 0.5) * 2
        
        # Correct vs incorrect predictions
        correct = (predictions == targets)
        
        # Confidence statistics
        analysis = {
            'avg_confidence': float(np.mean(confidence)),
            'avg_confidence_correct': float(np.mean(confidence[correct])),
            'avg_confidence_incorrect': float(np.mean(confidence[~correct])),
            'high_confidence_correct': int(np.sum((confidence > 0.8) & correct)),
            'high_confidence_incorrect': int(np.sum((confidence > 0.8) & (~correct))),
            'low_confidence_correct': int(np.sum((confidence < 0.2) & correct)),
            'low_confidence_incorrect': int(np.sum((confidence < 0.2) & (~correct)))
        }
        
        # Confidence bins
        bins = np.linspace(0, 1, 11)
        confidence_bins = np.digitize(confidence, bins) - 1
        
        bin_accuracy = []
        bin_counts = []
        
        for i in range(10):
            mask = confidence_bins == i
            if np.sum(mask) > 0:
                bin_accuracy.append(float(np.mean(correct[mask])))
                bin_counts.append(int(np.sum(mask)))
            else:
                bin_accuracy.append(0.0)
                bin_counts.append(0)
        
        analysis['confidence_bins'] = {
            'bin_edges': bins.tolist(),
            'bin_accuracy': bin_accuracy,
            'bin_counts': bin_counts
        }
        
        return analysis
    
    def _analyze_by_length(self, 
                         predictions: np.ndarray, 
                         targets: np.ndarray, 
                         lengths: np.ndarray) -> Dict[str, Any]:
        """
        Analyze performance by sequence length.
        
        Args:
            predictions: Model predictions
            targets: True targets
            lengths: Sequence lengths
            
        Returns:
            Length analysis results
        """
        # Define length bins
        length_percentiles = np.percentile(lengths, [25, 50, 75])
        
        short_mask = lengths <= length_percentiles[0]
        medium_mask = (lengths > length_percentiles[0]) & (lengths <= length_percentiles[2])
        long_mask = lengths > length_percentiles[2]
        
        analysis = {
            'length_percentiles': length_percentiles.tolist(),
            'short_sequences': {
                'count': int(np.sum(short_mask)),
                'accuracy': float(accuracy_score(targets[short_mask], predictions[short_mask])),
                'avg_length': float(np.mean(lengths[short_mask]))
            },
            'medium_sequences': {
                'count': int(np.sum(medium_mask)),
                'accuracy': float(accuracy_score(targets[medium_mask], predictions[medium_mask])),
                'avg_length': float(np.mean(lengths[medium_mask]))
            },
            'long_sequences': {
                'count': int(np.sum(long_mask)),
                'accuracy': float(accuracy_score(targets[long_mask], predictions[long_mask])),
                'avg_length': float(np.mean(lengths[long_mask]))
            }
        }
        
        return analysis
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray, 
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix to plot
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, 
                      fpr: np.ndarray, 
                      tpr: np.ndarray, 
                      auc: float,
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc: AUC score
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_analysis(self, 
                               confidence_analysis: Dict[str, Any],
                               save_path: Optional[str] = None) -> None:
        """
        Plot confidence analysis.
        
        Args:
            confidence_analysis: Confidence analysis results
            save_path: Path to save the plot (optional)
        """
        bins_data = confidence_analysis['confidence_bins']
        bin_centers = [(bins_data['bin_edges'][i] + bins_data['bin_edges'][i+1]) / 2 
                      for i in range(len(bins_data['bin_edges']) - 1)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy by confidence
        ax1.bar(bin_centers, bins_data['bin_accuracy'], width=0.08, alpha=0.7)
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Confidence Level')
        ax1.grid(True, alpha=0.3)
        
        # Count by confidence
        ax2.bar(bin_centers, bins_data['bin_counts'], width=0.08, alpha=0.7, color='orange')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Sample Count by Confidence Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ComprehensiveEvaluator:
    """
    Evaluator for comparing multiple models.
    """
    
    def __init__(self, 
                 models: Dict[str, nn.Module], 
                 device: Optional[torch.device] = None):
        """
        Initialize comprehensive evaluator.
        
        Args:
            models: Dictionary of models to evaluate
            device: Device to run evaluation on
        """
        self.models = models
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluators = {
            name: ModelEvaluator(model, self.device) 
            for name, model in models.items()
        }
    
    def evaluate_all_models(self, 
                          data_loaders: Dict[str, DataLoader]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models.
        
        Args:
            data_loaders: Dictionary of data loaders for each model type
            
        Returns:
            Dictionary with evaluation results for each model
        """
        results = {}
        
        for model_name, evaluator in self.evaluators.items():
            print(f"Evaluating {model_name}...")
            
            # Get appropriate data loader
            if 'lstm' in model_name:
                data_loader = data_loaders['lstm']
            else:
                data_loader = data_loaders['verbalizer']
            
            # Evaluate model
            results[model_name] = evaluator.detailed_evaluation(data_loader)
        
        return results
    
    def compare_models(self, 
                      results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare model results.
        
        Args:
            results: Evaluation results for all models
            
        Returns:
            Comparison analysis
        """
        comparison = {
            'metrics_comparison': {},
            'best_model': {},
            'model_rankings': {}
        }
        
        # Extract metrics for comparison
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for metric in metrics_names:
            comparison['metrics_comparison'][metric] = {
                model_name: results[model_name]['metrics'][metric]
                for model_name in results.keys()
            }
            
            # Find best model for this metric
            best_model = max(
                comparison['metrics_comparison'][metric].items(),
                key=lambda x: x[1]
            )
            comparison['best_model'][metric] = {
                'model': best_model[0],
                'score': best_model[1]
            }
        
        # Overall ranking (based on F1 score)
        f1_scores = comparison['metrics_comparison']['f1']
        ranked_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['model_rankings']['by_f1'] = ranked_models
        
        return comparison
    
    def plot_model_comparison(self, 
                            comparison: Dict[str, Any],
                            save_path: Optional[str] = None) -> None:
        """
        Plot model comparison.
        
        Args:
            comparison: Comparison results
            save_path: Path to save the plot (optional)
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        model_names = list(comparison['metrics_comparison']['accuracy'].keys())
        
        # Prepare data for plotting
        data = []
        for metric in metrics:
            for model_name in model_names:
                data.append({
                    'Model': model_name,
                    'Metric': metric.upper(),
                    'Score': comparison['metrics_comparison'][metric][model_name]
                })
        
        # Create DataFrame for seaborn
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Metric', y='Score', hue='Model')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, 
                                 results: Dict[str, Dict[str, Any]],
                                 comparison: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            comparison: Comparison analysis
            save_path: Path to save the report (optional)
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall comparison
        report.append("OVERALL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        report.append(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'AUC':<10}")
        report.append("-" * 40)
        
        for model_name in results.keys():
            metrics = results[model_name]['metrics']
            report.append(f"{model_name:<20} {metrics['accuracy']:<10.4f} "
                         f"{metrics['f1']:<10.4f} {metrics['auc']:<10.4f}")
        
        report.append("")
        
        # Best models by metric
        report.append("BEST MODELS BY METRIC")
        report.append("-" * 30)
        for metric, best in comparison['best_model'].items():
            report.append(f"{metric.upper()}: {best['model']} ({best['score']:.4f})")
        
        report.append("")
        
        # Detailed results for each model
        for model_name, result in results.items():
            report.append(f"DETAILED RESULTS: {model_name.upper()}")
            report.append("-" * 50)
            
            # Basic metrics
            metrics = result['metrics']
            report.append(f"Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"Precision: {metrics['precision']:.4f}")
            report.append(f"Recall: {metrics['recall']:.4f}")
            report.append(f"F1 Score: {metrics['f1']:.4f}")
            report.append(f"AUC: {metrics['auc']:.4f}")
            report.append("")
            
            # Confidence analysis
            if 'confidence_analysis' in result:
                conf = result['confidence_analysis']
                report.append("Confidence Analysis:")
                report.append(f"  Average Confidence: {conf['avg_confidence']:.4f}")
                report.append(f"  High Confidence Correct: {conf['high_confidence_correct']}")
                report.append(f"  High Confidence Incorrect: {conf['high_confidence_incorrect']}")
                report.append("")
            
            # Length analysis
            if result['length_analysis']:
                length = result['length_analysis']
                report.append("Performance by Sequence Length:")
                report.append(f"  Short sequences: {length['short_sequences']['accuracy']:.4f}")
                report.append(f"  Medium sequences: {length['medium_sequences']['accuracy']:.4f}")
                report.append(f"  Long sequences: {length['long_sequences']['accuracy']:.4f}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
