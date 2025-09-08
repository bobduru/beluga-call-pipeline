import os
import json
import numpy as np
import torch


def aggregate_folds_testing_metrics(directory: str) -> None:
    """
    Aggregate testing metrics across all folds in a directory, including other_tests and model sizes.
    
    Args:
        directory: Path to the directory containing fold_* subdirectories
    """
    # Find all fold directories
    fold_dirs = [d for d in os.listdir(directory) if d.startswith('fold_')]
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {directory}")

    # Initialize metrics containers
    main_metrics = {}
    other_tests_metrics = {}
    model_info = None  # Will store model size info from the last fold
    
    # Process each fold
    for fold_dir in fold_dirs:
        details_path = os.path.join(directory, fold_dir, 'training_details.json')
        if not os.path.exists(details_path):
            continue
            
        with open(details_path, 'r') as f:
            fold_data = json.load(f)
            
        # Store model info from the last fold
        model_info = {
            'model_size_mb': fold_data.get('model_size_mb'),
            'model_size_kb': fold_data.get('model_size_kb'),
            'quantized_model_size_mb': fold_data.get('quantized_model_size_mb'),
            'quantized_model_size_kb': fold_data.get('quantized_model_size_kb')
        }
        
        # Process main testing metrics
        testing_metrics = fold_data.get('testing_metrics', {})
        for metric_name, value in testing_metrics.items():
            if metric_name in ['time_elapsed', 'loss']:
                continue
                
            if metric_name not in main_metrics:
                main_metrics[metric_name] = {label: [] for label in value.keys()}
            
            for label, val in value.items():
                main_metrics[metric_name][label].append(val)
        
        # Process other_tests metrics
        other_tests = fold_data.get('other_tests', {})
        for test_name, test_metrics in other_tests.items():
            if test_name not in other_tests_metrics:
                other_tests_metrics[test_name] = {}
                
            for metric_name, value in test_metrics.items():
                if metric_name in ['time_elapsed', 'loss']:
                    continue
                    
                if metric_name not in other_tests_metrics[test_name]:
                    other_tests_metrics[test_name][metric_name] = {label: [] for label in value.keys()}
                
                for label, val in value.items():
                    other_tests_metrics[test_name][metric_name][label].append(val)

    # Aggregate main metrics
    aggregated_main_metrics = {
        metric: {
            label: {
                'values': values,
                'mean': float(np.mean(values)) if metric != 'confusion_matrix' else None,
                'std': float(np.std(values)) if metric != 'confusion_matrix' else None
            }
            for label, values in label_values.items()
        }
        for metric, label_values in main_metrics.items()
    }

    # Aggregate other_tests metrics
    aggregated_other_tests = {
        test_name: {
            metric: {
                label: {
                    'values': values,
                    'mean': float(np.mean(values)) if metric != 'confusion_matrix' else None,
                    'std': float(np.std(values)) if metric != 'confusion_matrix' else None
                }
                for label, values in label_values.items()
            }
            for metric, label_values in test_metrics.items()
        }
        for test_name, test_metrics in other_tests_metrics.items()
    }

    # Combine all results
    final_results = {
        'main_results': aggregated_main_metrics,
        'other_tests': aggregated_other_tests,
        'model_info': model_info
    }
    
    # Save results
    output_path = os.path.join(directory, 'cross_val_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)

import torch

def size_of_model(model, log=True):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p")/1e6
    
    size_kb =  os.path.getsize("temp.p") / 1024
    if log :
        print('Size (MB):', size_mb)
        print('Size (KB):', os.path.getsize("temp.p") / 1024)
    os.remove('temp.p')
    return size_mb, size_kb

def print_model_parameter_summary(model):
    print(f"{'Layer':<60} {'Params':>12}")
    print("-" * 75)
    total = 0
    module_totals = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"{name:<60} {num_params:>12,}")
            total += num_params

            # Get the top-level module (e.g., 'features', 'heads.ECHO', etc.)
            prefix = name.split('.')[0]
            # For heads, group by head name (e.g., 'heads.ECHO')
            if prefix == "heads":
                head_name = ".".join(name.split('.')[:2])
                module_totals.setdefault(head_name, 0)
                module_totals[head_name] += num_params
            else:
                module_totals.setdefault(prefix, 0)
                module_totals[prefix] += num_params

    print("-" * 75)
    print(f"{'Total Trainable Params':<60} {total:>12,}")
    print("\nParameter count by top-level module:")
    for module, count in module_totals.items():
        print(f"{module:<20}: {count:,}")


def save_model(model, path, **metadata):
    """
    Save a model's state_dict and any additional metadata.

    Args:
        model (torch.nn.Module): the model to save.
        path (str): file path to save the model (e.g., 'checkpoints/model.pt').
        **metadata: arbitrary keyword arguments (e.g., n_layers=6, num_classes=4).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        **metadata
    }
    torch.save(checkpoint, path)


def make_multilabel_metrics_readable(metrics_obj, labels_mapping):
    new_metrics_obj = {}

    for metric_name, value in metrics_obj.items():
        multilabel_metric = metric_name.startswith('multilabel_')
        metric_name = metric_name.replace('multilabel_', '') if multilabel_metric else metric_name

        if metric_name == "time_elapsed" or metric_name =="loss":
            new_metrics_obj[metric_name] = value
            continue
        
        if metric_name not in new_metrics_obj:
            new_metrics_obj[metric_name] = {}

        if not multilabel_metric:
            new_metrics_obj[metric_name]["Labels_Average"] = value
        else:
            for label_idx, val in enumerate(value):
                if label_idx < len(labels_mapping):
                    label = labels_mapping[label_idx]
                    new_metrics_obj[metric_name][label] = val

    return new_metrics_obj

def get_model_size(model):
    size = sum(p.numel() for p in model.parameters())
    return size

def get_best_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    return device