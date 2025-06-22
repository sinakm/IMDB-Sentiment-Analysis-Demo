"""
Compare LSTM and Verbalizer models by running them individually and comparing results.

This script runs each model training script and compares the results.

Usage:
    python compare_all_models.py --quick    # Quick comparison (fewer epochs)
    python compare_all_models.py --full     # Full comparison (more epochs)
"""

import argparse
import subprocess
import json
import time
import sys
from pathlib import Path
import pandas as pd


def run_model_training(script_name, args_dict):
    """Run a model training script with given arguments."""
    print(f"\n{'='*60}")
    print(f"RUNNING {script_name.upper()}")
    print(f"{'='*60}")
    
    # Build command using current Python interpreter (from virtual environment)
    cmd = [sys.executable, script_name]
    for key, value in args_dict.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        elif not isinstance(value, bool):
            cmd.extend([f'--{key}', str(value)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the script
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        execution_time = time.time() - start_time
        print(f"✅ {script_name} completed successfully in {execution_time:.1f} seconds")
        return True, execution_time, result.stdout
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        print(f"❌ {script_name} failed after {execution_time:.1f} seconds")
        print(f"Error: {e.stderr}")
        return False, execution_time, e.stderr


def load_results():
    """Load results from both model training runs."""
    results = {}
    
    # Try to load each result file
    result_files = {
        'verbalizer': 'results/verbalizer_only_results.json',
        'lstm': 'results/lstm_only_results.json'
    }
    
    for model_name, file_path in result_files.items():
        try:
            with open(file_path, 'r') as f:
                results[model_name] = json.load(f)
            print(f"✅ Loaded results for {model_name}")
        except FileNotFoundError:
            print(f"❌ Results file not found for {model_name}: {file_path}")
            results[model_name] = None
    
    return results


def create_comparison_table(results):
    """Create a comparison table from results."""
    comparison_data = []
    
    for model_name, result in results.items():
        if result is not None:
            comparison_data.append({
                'Model': model_name.title(),
                'Accuracy (%)': f"{result['test_accuracy']*100:.2f}",
                'F1 Score': f"{result['test_f1']:.4f}",
                'Training Time (min)': f"{result['training_time_minutes']:.1f}",
                'Trainable Params': f"{result['model_info']['trainable_parameters']:,}",
                'Total Params': f"{result['model_info']['total_parameters']:,}",
                'Approach': 'Pre-computed embeddings' if model_name == 'verbalizer' else 'Learned embeddings'
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        return df
    else:
        return None


def print_detailed_comparison(results):
    """Print detailed comparison of both models."""
    print("\n" + "="*80)
    print("DETAILED MODEL COMPARISON: LSTM vs VERBALIZER")
    print("="*80)
    
    for model_name, result in results.items():
        if result is not None:
            print(f"\n{model_name.upper()} RESULTS:")
            print(f"  Test Accuracy: {result['test_accuracy']*100:.2f}%")
            print(f"  F1 Score: {result['test_f1']:.4f}")
            print(f"  Precision: {result['test_precision']:.4f}")
            print(f"  Recall: {result['test_recall']:.4f}")
            print(f"  Training Time: {result['training_time_minutes']:.1f} minutes")
            print(f"  Trainable Parameters: {result['model_info']['trainable_parameters']:,}")
            print(f"  Total Parameters: {result['model_info']['total_parameters']:,}")
            
            # Model-specific info
            if model_name == 'verbalizer':
                print(f"  Approach: Pre-computed ModernBERT embeddings")
                print(f"  Architecture: {result['model_info']['description']}")
                print(f"  Embedding Dim: {result['model_info']['embedding_dim']}")
            elif model_name == 'lstm':
                print(f"  Approach: Learned embeddings from scratch")
                print(f"  Vocabulary Size: {result['model_info']['vocab_size']:,}")
                print(f"  Hidden Dim: {result['model_info']['hidden_dim']}")
                print(f"  Bidirectional: {result['model_info']['bidirectional']}")
                print(f"  Attention: {result['model_info']['use_attention']}")


def print_comparison_insights(results):
    """Print insights comparing the two approaches."""
    print(f"\n{'='*80}")
    print("COMPARISON INSIGHTS: TRADITIONAL vs MODERN APPROACH")
    print(f"{'='*80}")
    
    if all(r is not None for r in results.values()):
        lstm_result = results['lstm']
        verb_result = results['verbalizer']
        
        print(f"\n🏗️  ARCHITECTURE COMPARISON:")
        print(f"  LSTM: Traditional RNN with learned embeddings")
        print(f"    • Learns everything from scratch")
        print(f"    • {lstm_result['model_info']['trainable_parameters']:,} trainable parameters")
        print(f"    • Bidirectional LSTM with attention mechanism")
        
        print(f"  Verbalizer: Modern approach with pre-trained embeddings")
        print(f"    • Leverages pre-trained ModernBERT knowledge")
        print(f"    • {verb_result['model_info']['trainable_parameters']:,} trainable parameters")
        print(f"    • Simple classifier on rich embeddings")
        
        print(f"\n⚡ EFFICIENCY COMPARISON:")
        lstm_time = lstm_result['training_time_minutes']
        verb_time = verb_result['training_time_minutes']
        speedup = lstm_time / verb_time if verb_time > 0 else 0
        print(f"  LSTM Training Time: {lstm_time:.1f} minutes")
        print(f"  Verbalizer Training Time: {verb_time:.1f} minutes")
        print(f"  Speedup: {speedup:.1f}x faster with verbalizer")
        
        print(f"\n🎯 PERFORMANCE COMPARISON:")
        lstm_acc = lstm_result['test_accuracy'] * 100
        verb_acc = verb_result['test_accuracy'] * 100
        acc_diff = verb_acc - lstm_acc
        print(f"  LSTM Accuracy: {lstm_acc:.2f}%")
        print(f"  Verbalizer Accuracy: {verb_acc:.2f}%")
        print(f"  Difference: {acc_diff:+.2f}% {'(Verbalizer wins)' if acc_diff > 0 else '(LSTM wins)' if acc_diff < 0 else '(Tie)'}")
        
        print(f"\n🔧 PARAMETER EFFICIENCY:")
        lstm_params = lstm_result['model_info']['trainable_parameters'] / 1_000_000
        verb_params = verb_result['model_info']['trainable_parameters'] / 1_000_000
        lstm_eff = lstm_result['test_accuracy'] / lstm_params if lstm_params > 0 else 0
        verb_eff = verb_result['test_accuracy'] / verb_params if verb_params > 0 else 0
        print(f"  LSTM: {lstm_eff:.3f} accuracy per million parameters")
        print(f"  Verbalizer: {verb_eff:.3f} accuracy per million parameters")
        
        print(f"\n💡 TRADE-OFFS:")
        print(f"  LSTM Advantages:")
        print(f"    • Self-contained (no external dependencies)")
        print(f"    • Learns task-specific representations")
        print(f"    • Traditional, well-understood approach")
        
        print(f"  Verbalizer Advantages:")
        print(f"    • Much faster training ({speedup:.1f}x speedup)")
        print(f"    • Leverages pre-trained knowledge")
        print(f"    • Fewer parameters to train")
        print(f"    • Modern transfer learning approach")


def save_comparison_results(results, comparison_df):
    """Save comparison results to file."""
    # Save detailed comparison
    comparison_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'comparison_type': 'LSTM vs Verbalizer',
        'individual_results': results,
        'summary': {}
    }
    
    if comparison_df is not None:
        comparison_summary['summary'] = comparison_df.to_dict('records')
    
    # Save to file
    output_file = Path('results/lstm_vs_verbalizer_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    print(f"\n📊 Comparison results saved to {output_file}")
    
    # Also save CSV table
    if comparison_df is not None:
        csv_file = Path('results/lstm_vs_verbalizer_comparison.csv')
        comparison_df.to_csv(csv_file, index=False)
        print(f"📊 Comparison table saved to {csv_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare LSTM vs Verbalizer Models')
    
    # Comparison mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick comparison (fewer epochs, smaller dataset)')
    parser.add_argument('--full', action='store_true',
                       help='Full comparison (more epochs, full dataset)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, just compare existing results')
    
    args = parser.parse_args()
    
    # Set default to quick if neither specified
    if not args.quick and not args.full:
        args.quick = True
    
    print("="*80)
    print("LSTM vs VERBALIZER COMPARISON PIPELINE")
    print("="*80)
    
    if args.quick:
        print("Mode: QUICK COMPARISON")
        common_args = {
            'num_epochs': 3,
            'batch_size': 32,
            'max_length': 256,
            'max_samples': 1000  # Quick test with 1K samples
        }
    else:
        print("Mode: FULL COMPARISON")
        common_args = {
            'num_epochs': 5,
            'batch_size': 32,
            'max_length': 256
            # No max_samples = full dataset
        }
    
    if not args.skip_training:
        print(f"Training parameters: {common_args}")
        
        # Model-specific arguments
        model_configs = {
            'train_lstm_only.py': {
                **common_args,
                'learning_rate': 1e-3,
                'hidden_dim': 128,
                'embed_dim': 200
            },
            'train_verbalizer_only.py': {
                **common_args,
                'learning_rate': 1e-3,
                'hidden_dim': 128
            }
        }
        
        # Run each model training
        training_results = {}
        total_start_time = time.time()
        
        for script, config in model_configs.items():
            success, exec_time, output = run_model_training(script, config)
            training_results[script] = {
                'success': success,
                'execution_time': exec_time,
                'output': output
            }
        
        total_time = time.time() - total_start_time
        print(f"\n🏁 All training completed in {total_time/60:.1f} minutes")
        
        # Print training summary
        print(f"\nTraining Summary:")
        for script, result in training_results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"  {script}: {status} ({result['execution_time']:.1f}s)")
    
    else:
        print("Skipping training, loading existing results...")
    
    # Load and compare results
    print(f"\n{'='*60}")
    print("LOADING AND COMPARING RESULTS")
    print(f"{'='*60}")
    
    results = load_results()
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    if comparison_df is not None:
        print(f"\n📊 LSTM vs VERBALIZER COMPARISON TABLE:")
        print(comparison_df.to_string(index=False))
    else:
        print("❌ No results available for comparison")
    
    # Print detailed comparison
    print_detailed_comparison(results)
    
    # Print comparison insights
    print_comparison_insights(results)
    
    # Save results
    save_comparison_results(results, comparison_df)
    
    print(f"\n🎉 LSTM vs Verbalizer comparison completed!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"\n💡 Key Takeaways:")
    print(f"  • LSTM: Traditional approach, learns from scratch")
    print(f"  • Verbalizer: Modern approach, leverages pre-trained knowledge")
    print(f"  • Both approaches offer different trade-offs in speed vs interpretability")


if __name__ == "__main__":
    main()
