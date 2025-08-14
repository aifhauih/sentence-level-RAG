from tools import llama_tools, experiment_tools
from tools.new_prompt_tools import PromptTools, RAGContextOptimizer
import json
import argparse
import datetime

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--num_calls", type=int, default=5)
    parser.add_argument("--tops", type=int, default=1)
    parser.add_argument("--tails", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--dataset_name", type=str, 
                       choices=['19', '20', '21', '22', 'dev_small', 'nq_test', 'hotpotqa_dev'])
    parser.add_argument("--retriever", type=str, default='bm25',
                       choices=['bm25', 'mt5', 'tct', 'e5', 'oracle', 'reverse_oracle'])
    parser.add_argument("--long_answer", type=str, default='True', choices=['False', 'True'])
    parser.add_argument("--optimization_strategy", type=str, default='balanced',
                       choices=['relevance', 'diversity', 'consistency', 'balanced', 'custom', 'none'])
    parser.add_argument("--full_permutation", type=str, default='False', choices=['False', 'True'])
    parser.add_argument("--max_candidates", type=int, default=20,
                       help="Maximum number of candidate documents to consider from retrieval results")
    
    # New: Support for custom alpha/beta parameters
    parser.add_argument("--alpha", type=float, default=None,
                       help="Relevance weight (only used when optimization_strategy='custom')")
    parser.add_argument("--beta", type=float, default=None,
                       help="Diversity weight (only used when optimization_strategy='custom')")
    
    args = parser.parse_args()
    
    k = args.k
    step = args.step
    num_calls = args.num_calls
    tops = args.tops
    tails = args.tails
    temperature = args.temperature
    dataset_name = args.dataset_name
    retriever_name = args.retriever
    long_answer = True if args.long_answer == 'True' else False
    optimization_strategy = args.optimization_strategy
    full_permutation = eval(args.full_permutation)
    max_candidates = args.max_candidates
    alpha = args.alpha
    beta = args.beta
    
    short_answer_identifier = 'random' if long_answer else 'short'
    
    print(f"Optimization strategy: {optimization_strategy}")
    if optimization_strategy == 'custom':
        print(f"Custom parameters: alpha={alpha}, beta={beta}, gamma={1-(alpha or 0)-(beta or 0)}")
    print(f"Full permutation: {full_permutation}")
    print(f"Max candidate documents: {max_candidates}")
    
    # Validate custom parameters
    if optimization_strategy == 'custom':
        if alpha is None or beta is None:
            print("Error: Must provide both alpha and beta parameters when using custom strategy")
            exit(1)
        if alpha < 0 or beta < 0 or alpha + beta > 1:
            print(f"Error: Invalid parameters alpha={alpha}, beta={beta}")
            print("Requirements: alpha >= 0, beta >= 0, alpha + beta <= 1")
            exit(1)
    
    # Load model and data
    llm = llama_tools.load_llama()
    
    # Initialize enhanced prompt tools
    enhanced_prompt_tools = EnhancedPromptTools()
    doc_dict, queries, res = enhanced_prompt_tools.prepare_data(dataset_name, retriever_name)
    
    # Set filename
    if optimization_strategy == 'none':
        # Use original logic
        file_suffix = 'prompt1' if long_answer else 'concise'
    elif optimization_strategy == 'custom':
        # Custom strategy filename
        file_suffix = f'custom_a{alpha}_b{beta}'
        if full_permutation:
            file_suffix += '_fullperm'
    else:
        # Use predefined optimization strategy
        file_suffix = f'optimized_{optimization_strategy}'
        if full_permutation:
            file_suffix += '_fullperm'
    
    file_name = f'./gen_results/{short_answer_identifier}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_{file_suffix}.json'
    
    # Record experiment settings
    setting_file_name = file_name.replace('.json', '_settings.json')
    setting_record = {
        'k': k, 'num_calls': num_calls, 'step': step, 'tops': tops, 'tails': tails,
        'temperature': temperature, 'query_set': dataset_name, 'retriever': retriever_name,
        'long_answer': long_answer, 'optimization_strategy': optimization_strategy,
        'full_permutation': full_permutation, 'max_candidates': max_candidates,
        'alpha': alpha, 'beta': beta, 'gamma': 1-(alpha or 0)-(beta or 0) if alpha is not None and beta is not None else None,
        'experiment_start_at': str(datetime.datetime.now())
    }
    
    with open(setting_file_name, "w+", encoding='UTF-8') as f:
        json.dump(setting_record, f, indent=4)
    
    # Read existing results
    try:
        with open(file_name, "r") as f:
            result_to_write = json.load(f)
            existed_qids_list = list(result_to_write.keys())
            existed_qids = len(result_to_write)
    except:
        result_to_write = {}
        existed_qids_list = []
        existed_qids = 0
    
    print(f"Processed queries: {existed_qids}")
    
    # Main experiment loop
    q_no = 0
    for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
        print(f'Query ID={q_no}--{qid}')
        
        if str(qid) in existed_qids_list:
            print("Already generated, skipping!")
            q_no += 1
            continue
        
        q_no += 1
        varying_context_result = {}
        
        if optimization_strategy == 'none':
            # Use original context composition logic
            from tools.prompt_tools import compose_context
            start_records, context_book = compose_context(
                qid=qid, res=res, k=k, step=step, tops=tops, tails=tails, 
                doc_dict=doc_dict, reverse_order=False
            )
        else:
            # Use optimized context selection
            if full_permutation:
                start_records, context_book = enhanced_prompt_tools.compose_context_with_optimization_and_permutations(
                    query=query, res=res, qid=qid, k=k, doc_dict=doc_dict,
                    optimization_strategy=optimization_strategy, 
                    full_permutations=full_permutation,
                    max_candidates=max_candidates,
                    alpha=alpha, beta=beta
                )
            else:
                start_records, context_book = enhanced_prompt_tools.compose_optimized_context(
                    query=query, res=res, qid=qid, k=k, doc_dict=doc_dict,
                    optimization_strategy=optimization_strategy,
                    max_candidates=max_candidates,
                    alpha=alpha, beta=beta
                )
        
        print('Starting records: ', start_records)
        
        # Generate for each context variant
        for start, context in zip(start_records, context_book):
            llm.set_seed(1000)
            print(f'\tStarting rank.{start}')
            
            # Assemble prompt
            from tools.enhanced_prompt_tools import prompt_assembler
            prompt = prompt_assembler(context, query, long_answer)
            print(f"Prompt length: {len(prompt)} characters")
            print(f"Context preview: {context[:200]}...")
            
            multi_call_results = {}
            for j in range(num_calls):
                print(f'\t\tCall #{j}')
                result = llama_tools.single_call(
                    llm=llm, prompt=prompt, temperature=temperature, long_answer=long_answer
                )
                multi_call_results.update({j: result})
            
            varying_context_result.update({start: multi_call_results})
        
        result_to_write.update({qid: varying_context_result})
        experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)

    # Print the full generated JSON file
    print("\n" + "="*50)
    print("Complete Generated JSON Output")
    print("="*50)
    print(json.dumps(result_to_write, indent=4))
    print("="*50 + "\n")
    
    del llm
    print("Experiment completed!")
    print(f"Results file: {file_name}")
    print(f"Settings file: {setting_file_name}")


# Script to compare different optimization strategies
def run_optimization_comparison(dataset_name='dev_small', k=3, num_calls=2):
    """
    Run comparison experiments for different optimization strategies
    """
    import subprocess
    import os
    
    # Predefined strategies
    strategies = ['relevance', 'diversity', 'consistency', 'balanced']
    
    # Custom alpha/beta combinations
    custom_params = [
        (0.8, 0.1),  # High relevance
        (0.4, 0.5),  # High diversity (negative)
        (0.2, 0.1),  # High consistency (gamma=0.7)
        (0.6, 0.2),  # Another balanced approach
    ]
    
    results_summary = []
    
    print(f"\n{'='*60}")
    print(f"Running optimization strategy comparison experiment")
    print(f"Dataset: {dataset_name}, k={k}, num_calls={num_calls}")
    print(f"{'='*60}")
    
    # Run predefined strategies
    for strategy in strategies:
        print(f"\nRunning strategy: {strategy}")
        cmd = [
            "python", "k-shot-new.py",
            "--dataset_name", dataset_name,
            "--k", str(k),
            "--num_calls", str(num_calls),
            "--optimization_strategy", strategy
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                print(f" Strategy {strategy} completed")
                results_summary.append(f" {strategy}: Success")
            else:
                print(f" Strategy {strategy} failed: {result.stderr}")
                results_summary.append(f" {strategy}: Failed")
        except subprocess.TimeoutExpired:
            print(f" Strategy {strategy} timed out")
            results_summary.append(f" {strategy}: Timeout")
        except Exception as e:
            print(f" Strategy {strategy} error: {e}")
            results_summary.append(f" {strategy}: Error")
    
    # Run custom parameter strategies
    for i, (alpha, beta) in enumerate(custom_params):
        strategy_name = f"custom_{i+1}"
        print(f"\nRunning custom strategy: alpha={alpha}, beta={beta}")
        
        cmd = [
            "python", "k-shot-new.py",
            "--dataset_name", dataset_name,
            "--k", str(k),
            "--num_calls", str(num_calls),
            "--optimization_strategy", "custom",
            "--alpha", str(alpha),
            "--beta", str(beta)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                print(f" Custom strategy {strategy_name} completed")
                results_summary.append(f" {strategy_name} (α={alpha},β={beta}): Success")
            else:
                print(f" Custom strategy {strategy_name} failed: {result.stderr}")
                results_summary.append(f" {strategy_name}: Failed")
        except subprocess.TimeoutExpired:
            print(f" Custom strategy {strategy_name} timed out")
            results_summary.append(f" {strategy_name}: Timeout")
        except Exception as e:
            print(f" Custom strategy {strategy_name} error: {e}")
            results_summary.append(f" {strategy_name}: Error")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Experiment summary:")
    print(f"{'='*60}")
    for summary in results_summary:
        print(summary)
    
    return results_summary


def analyze_optimization_results(dataset_name='dev_small', k=3):
    """
    Analyze results from different optimization strategies
    """
    import glob
    import json
    
    print(f"\n{'='*60}")
    print(f"Analyzing optimization results")
    print(f"{'='*60}")
    
    # Find relevant result files
    pattern = f"./gen_results/*_{k}shot_*_{dataset_name}_*.json"
    result_files = glob.glob(pattern)
    
    optimization_files = [f for f in result_files if 'optimized' in f or 'custom' in f]
    
    if not optimization_files:
        print("No optimization strategy result files found")
        return
    
    print(f"Found {len(optimization_files)} optimization result files:")
    for f in optimization_files:
        print(f"  - {os.path.basename(f)}")
    
    # Simple result statistics
    for file_path in optimization_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            strategy_name = os.path.basename(file_path).split('_')[4:7]  # Extract strategy name
            num_queries = len(results)
            
            print(f"\nStrategy: {'-'.join(strategy_name)}")
            print(f"  Processed queries: {num_queries}")
            
            # Count context strategies per query
            if results:
                sample_qid = list(results.keys())[0]
                num_strategies = len(results[sample_qid])
                print(f"  Context strategies per query: {num_strategies}")
                
        except Exception as e:
            print(f"   Failed to read file: {e}")


if __name__ == "__main__":
    # If running this script directly, execute comparison experiment
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run comparison experiment
        run_optimization_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # Analyze results
        analyze_optimization_results()
    else:
        # Normal experiment flow (original logic)
        pass