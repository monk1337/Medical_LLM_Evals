import os
import json
import pandas as pd
from typing import Dict, List
import argparse

# Add this at the beginning of the script
parser = argparse.ArgumentParser(description='Process model results')
parser.add_argument('--modelname', type=str, help='Model name to use as prefix')
args = parser.parse_args()

def find_result_files(root_dir: str) -> List[tuple]:
    """
    Find all results JSON files and their corresponding checkpoint numbers.
    Returns list of (checkpoint_number, file_path) tuples.
    """
    result_files = []
    
    for root, dirs, files in os.walk(root_dir):
        if 'result' in root.lower():
            for file in files:
                if file.startswith('result') and file.endswith('.json'):
                    # Extract checkpoint number from path
                    checkpoint = None
                    path_parts = root.split(os.sep)
                    for part in path_parts:
                        if part.startswith('checkpoint-'):
                            checkpoint = part
                            break
                    
                    if checkpoint:
                        result_files.append((checkpoint, os.path.join(root, file)))
    
    return result_files

def extract_metrics(json_path: str) -> Dict[str, float]:
    """
    Extract specific accuracy metrics from a results JSON file and convert to percentages.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics = {}
    target_tasks = [
        'medmcqa',
        'medqa_4options',
        'mmlu_anatomy',
        'mmlu_clinical_knowledge',
        'mmlu_college_biology',
        'mmlu_college_medicine',
        'mmlu_medical_genetics',
        'mmlu_professional_medicine',
        'pubmedqa'
    ]
    
    results = data.get('results', {})
    for task in target_tasks:
        if task in results:
            # Convert to percentage and round to 3 decimal places
            value = results[task].get('acc,none')
            if value is not None:
                metrics[task] = round(value * 100, 3)
            else:
                metrics[task] = None
    
    # Calculate average of available metrics
    valid_metrics = [v for v in metrics.values() if v is not None]
    if valid_metrics:
        metrics['average'] = round(sum(valid_metrics) / len(valid_metrics), 3)
    else:
        metrics['average'] = None
    
    return metrics

def process_all_results(root_dir: str, output_file: str = 'model_metrics.csv', checkpoint_prefix: str = None):
    """
    Process all result files and create a CSV with metrics as percentages.
    
    Parameters:
        root_dir (str): Root directory to search for result files
        output_file (str): Output CSV filename
        checkpoint_prefix (str): Optional prefix to add before checkpoint numbers (e.g., "model_name_")
    """
    result_files = find_result_files(root_dir)
    all_metrics = []
    
    for checkpoint, file_path in result_files:
        metrics = extract_metrics(file_path)
        # Add prefix to checkpoint if provided
        if checkpoint_prefix:
            metrics['checkpoint'] = f"{checkpoint_prefix}{checkpoint}"
        else:
            metrics['checkpoint'] = checkpoint
        all_metrics.append(metrics)
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Reorder columns to put checkpoint and average first
        cols = ['checkpoint', 'average'] + [col for col in df.columns if col not in ['checkpoint', 'average']]
        df = df[cols]
        
        # Format float columns to 3 decimal places
        float_cols = [col for col in df.columns if col != 'checkpoint']
        for col in float_cols:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
            
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No result files found.")

# Usage examples
if __name__ == "__main__":
    root_directory = "."  # Replace with your root directory path
    
    # Example 1: Without prefix (original behavior)
    # process_all_results(root_directory)
    
    process_all_results(
        root_directory,
        output_file='model_metrics.csv',
        checkpoint_prefix=args.modelname)
