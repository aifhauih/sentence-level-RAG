import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_qrel_comparison(json_files, save_path):
    """
    Compare metrics (precision, recall, f1) average and max values for qrel_2 and qrel_3 
    across multiple JSON files and generate histograms.

    Args:
        json_files (list): List of JSON file paths to process
    """
    data = {}

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_path in json_files:
        with open(file_path, 'r') as f:
            try:
                json_data = json.load(f)
                
                for id_key, metrics_data in json_data.items():
                    # Skip empty entries
                    if not metrics_data:
                        continue
                    
                    for metric_name, values in metrics_data.items():
                        if metric_name not in data:
                            data[metric_name] = {'qrel_2': {'precision': {'avg': [], 'max': []},
                                                           'recall': {'avg': [], 'max': []},
                                                           'f1': {'avg': [], 'max': []}},
                                                 'qrel_3': {'precision': {'avg': [], 'max': []},
                                                           'recall': {'avg': [], 'max': []},
                                                           'f1': {'avg': [], 'max': []}}}

                        qrel_2_avg_values = {'precision': [], 'recall': [], 'f1': []}
                        qrel_2_max_values = {'precision': [], 'recall': [], 'f1': []}
                        qrel_3_avg_values = {'precision': [], 'recall': [], 'f1': []}
                        qrel_3_max_values = {'precision': [], 'recall': [], 'f1': []}
                        
                        for sub_id, qrel_values in values.items():
                            # Extract all metrics for qrel_2
                            qrel_2_avg_values['precision'].append(qrel_values['qrel_2']['precision']['avg'])
                            qrel_2_max_values['precision'].append(qrel_values['qrel_2']['precision']['max'])
                            qrel_2_avg_values['recall'].append(qrel_values['qrel_2']['recall']['avg'])
                            qrel_2_max_values['recall'].append(qrel_values['qrel_2']['recall']['max'])
                            qrel_2_avg_values['f1'].append(qrel_values['qrel_2']['f1']['avg'])
                            qrel_2_max_values['f1'].append(qrel_values['qrel_2']['f1']['max'])

                            # Extract all metrics for qrel_3
                            qrel_3_avg_values['precision'].append(qrel_values['qrel_3']['precision']['avg'])
                            qrel_3_max_values['precision'].append(qrel_values['qrel_3']['precision']['max'])
                            qrel_3_avg_values['recall'].append(qrel_values['qrel_3']['recall']['avg'])
                            qrel_3_max_values['recall'].append(qrel_values['qrel_3']['recall']['max'])
                            qrel_3_avg_values['f1'].append(qrel_values['qrel_3']['f1']['avg'])
                            qrel_3_max_values['f1'].append(qrel_values['qrel_3']['f1']['max'])

                        # Calculate final averages for each metric and add to data structure
                        data[metric_name]['qrel_2']['precision']['avg'].append(np.mean(qrel_2_avg_values['precision']))
                        data[metric_name]['qrel_2']['precision']['max'].append(np.mean(qrel_2_max_values['precision']))
                        data[metric_name]['qrel_2']['recall']['avg'].append(np.mean(qrel_2_avg_values['recall']))
                        data[metric_name]['qrel_2']['recall']['max'].append(np.mean(qrel_2_max_values['recall']))
                        data[metric_name]['qrel_2']['f1']['avg'].append(np.mean(qrel_2_avg_values['f1']))
                        data[metric_name]['qrel_2']['f1']['max'].append(np.mean(qrel_2_max_values['f1']))

                        data[metric_name]['qrel_3']['precision']['avg'].append(np.mean(qrel_3_avg_values['precision']))
                        data[metric_name]['qrel_3']['precision']['max'].append(np.mean(qrel_3_max_values['precision']))
                        data[metric_name]['qrel_3']['recall']['avg'].append(np.mean(qrel_3_avg_values['recall']))
                        data[metric_name]['qrel_3']['recall']['max'].append(np.mean(qrel_3_max_values['recall']))
                        data[metric_name]['qrel_3']['f1']['avg'].append(np.mean(qrel_3_avg_values['f1']))
                        data[metric_name]['qrel_3']['f1']['max'].append(np.mean(qrel_3_max_values['f1']))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    if not data:
        print("No valid data found for plotting.")
        return

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Define metrics and their colors
    metrics = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1-score']
    colors_avg = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    colors_max = ['#aec7e8', '#ffbb78', '#98df8a']  # Light blue, light orange, light green

    # Get category names
    categories = list(data.keys())
    x = np.arange(len(categories))  # Category positions on x-axis
    width = 0.15  # Bar width

    # Plot qrel_2 subplot
    ax1 = axes[0]
    for i, metric in enumerate(metrics):
        # Extract data
        avg_values = [data[cat]['qrel_2'][metric]['avg'][0] for cat in categories]
        max_values = [data[cat]['qrel_2'][metric]['max'][0] for cat in categories]
        
        # Plot average and max values
        ax1.bar(x - width + i * width * 2/3, avg_values, width, label=f'{metric_labels[i]} (Avg)', color=colors_avg[i])
        ax1.bar(x + width + i * width * 2/3, max_values, width, label=f'{metric_labels[i]} (Max)', color=colors_max[i])

    ax1.set_ylabel('Scores')
    ax1.set_title('qrel_2 Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot qrel_3 subplot
    ax2 = axes[1]
    for i, metric in enumerate(metrics):
        # Extract data
        avg_values = [data[cat]['qrel_3'][metric]['avg'][0] for cat in categories]
        max_values = [data[cat]['qrel_3'][metric]['max'][0] for cat in categories]
        
        # Plot average and max values
        ax2.bar(x - width + i * width * 2/3, avg_values, width, label=f'{metric_labels[i]} (Avg)', color=colors_avg[i])
        ax2.bar(x + width + i * width * 2/3, max_values, width, label=f'{metric_labels[i]} (Max)', color=colors_max[i])

    ax2.set_ylabel('Scores')
    ax2.set_title('qrel_3 Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path + '/result.png', dpi=400)
    plt.show()


# Call function with JSON file list
json_files_to_compare = ['./eval_results/random_answers_5shot_3calls_0_0_bm25_dl_19_sentence_level_eval.json', 
                         './eval_results/random_answers_5shot_3calls_1_0_bm25_dl_19_optimized_balanced_eval.json', 
                         './eval_results/random_answers_5shot_3calls_1_0_bm25_dl_19_prompt1_eval.json']

plot_qrel_comparison(json_files_to_compare, save_path="visual_results")