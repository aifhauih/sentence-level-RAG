from evaluate import load
import pandas as pd
import pickle
import json
import argparse
import os
import glob
from tools import eval_tools

import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def set_model_cache_dir(cache_dir):
    """set_model_cache_directory"""
    if cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        print(f"set model cache directory: {cache_dir}")


def prepare_qids_qrels_docdict(dataset_name):
    """prepare data for evaluation"""
    if dataset_name in ['21', '22']:
        with open('./doc_dicts/msmarco_passage_v2_dict.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
    else:
        with open('./doc_dicts/msmarco_passage_dict.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            
    print("dict loaded")
    
    queries = pd.read_csv(f'./queries/queries_{dataset_name}.csv')
    queries['qid'] = queries['qid'].astype('str')
    qids = queries.qid.tolist()
    
    if dataset_name in ['21', '22']:
        qrels = pd.read_csv('./qrels/qrels_v2.csv')
    elif dataset_name == 'dev_small':
        qrels = pd.read_csv('./qrels/qrels_dev.csv')
    else:
        qrels = pd.read_csv('./qrels/qrels.csv')
    
    print("data loaded")
    qrels['qid'] = qrels['qid'].astype('str')
    qrels['docno'] = qrels['docno'].astype('str')
    
    return qids, qrels, doc_dict


def prepare_qids_goldanswers(dataset_name):
    """prepare goldem answeer for Natural Questions"""
    queries = pd.read_csv(f'./queries/queries_{dataset_name}.csv')
    queries['qid'] = queries['qid'].astype('str')
    qids = queries.qid.tolist()
    nq_gds = pd.read_csv('./golden_answers/gdas_nq_test.csv')
    return qids, nq_gds


def get_docnos(qid, doc_dict, qrels):
    """Retrieve relevant documents and handle missing docno"""
    docnos_0, docnos_1, docnos_2, docnos_3 = [], [], [], []
    missing_docnos = []
    
    # handling documents with different relevance levels
    for label in [0, 1, 2, 3]:
        docnos = qrels[(qrels.qid == qid) & (qrels.label == label)].docno.tolist()
        doc_list = []
        for docno in docnos:
            if str(docno) in doc_dict:
                doc_list.append(doc_dict[str(docno)])
            else:
                missing_docnos.append(str(docno))
        
        if label == 0:
            docnos_0 = doc_list
        elif label == 1:
            docnos_1 = doc_list
        elif label == 2:
            docnos_2 = doc_list
        elif label == 3:
            docnos_3 = doc_list
    
    # save missing docno to file
    # if missing_docnos:
    #     with open(f'./missing_docnos_qid_{qid}.txt', 'w') as f:
    #         f.write('\n'.join(missing_docnos))
    
    docno_dict = {0: docnos_0, 1: docnos_1, 2: docnos_2, 3: docnos_3}
    return docno_dict


def evaluator(to_eval: str, docno_dict: dict, qrel_level: int, local_model_path: str = "bert-large-uncased"):
    """Using BERTScore to evaluate a single relevance level"""
    print(f'\t\t\t{qrel_level}')
    
    doc_texts = docno_dict[qrel_level]
    if len(doc_texts) == 0:
        r = {
            'precision': {'avg': -1, 'max': -1},
            'recall': {'avg': -1, 'max': -1},
            'f1': {'avg': -1, 'max': -1},
        }
        return r

    pred_text = to_eval
    predictions = [pred_text] * len(doc_texts)
    references = doc_texts
    
    # Using the specified model path
    results = answer_scorer.compute(
        predictions=predictions, 
        references=references, 
        lang="en", 
        model_type=local_model_path,
        verbose=False
    )

    precisions, recall, f1 = results['precision'], results['recall'], results['f1']

    r = {
        'precision': {'avg': sum(precisions)/len(precisions), 'max': max(precisions)},
        'recall': {'avg': sum(recall)/len(recall), 'max': max(recall)},
        'f1': {'avg': sum(f1)/len(f1), 'max': max(f1)},
    }
    return r


def eval_by_qrels(to_eval: str, docno_dict, qrel_levels=[2, 3], local_model_path: str = "bert-large-uncased"):
    """eval based on qrels"""
    r = {}
    for level in qrel_levels:
        r.update({f'qrel_{level}': evaluator(to_eval, docno_dict, level, local_model_path)})
    return r


def eval_by_goldanswers(to_eval: str, refs):
    """eval based on golden answers（EM/F1）"""
    r = {}
    try:
        r.update({'EM': eval_tools.cal_em(to_eval, refs)})
        r.update({'F1': eval_tools.cal_f1(to_eval, refs)})
    except:
        r.update({'EM': -1})
        r.update({'F1': -1})        
    return r


def find_generation_file(k, num_calls, tops, tails, retriever_name, dataset_name, 
                        optimization_strategy='balanced', alpha=None, beta=None, 
                        full_permutation=False, long_answer=True):
    """Locate the corresponding generated result file"""
    
    short_answer_identifier = 'random' if long_answer else 'short'
    
    # construct file suffix based on optimization strategy
    if optimization_strategy == 'none':
        file_suffix = 'prompt1' if long_answer else 'concise'
    elif optimization_strategy == 'custom' and alpha is not None and beta is not None:
        file_suffix = f'custom_a{alpha}_b{beta}'
        if full_permutation:
            file_suffix += '_fullperm'
    else:
        file_suffix = f'optimized_{optimization_strategy}'
        if full_permutation:
            file_suffix += '_fullperm'
    
    # build file name
    file_name = f'./gen_results/{short_answer_identifier}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_{file_suffix}.json'
    
    if os.path.exists(file_name):
        return file_name
    
    # if the file does not exist, attempt to find similar files
    pattern = f'./gen_results/{short_answer_identifier}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_*.json'
    possible_files = glob.glob(pattern)
    
    if possible_files:
        print(f"No exact matching file found: {file_name}")
        print("Possible matching files found:")
        for i, f in enumerate(possible_files):
            print(f"  {i+1}. {os.path.basename(f)}")
        
        return possible_files[0]
    
    return None


def auto_detect_file_type(file_path):
    """Automatically detect file type and parameters"""
    basename = os.path.basename(file_path)
    parts = basename.split('_')
    
    info = {
        'optimization_strategy': 'none',
        'alpha': None,
        'beta': None,
        'full_permutation': False
    }
    
    if 'optimized' in basename:
        # Search for strategy name
        for i, part in enumerate(parts):
            if part == 'optimized':
                if i + 1 < len(parts):
                    strategy = parts[i + 1]
                    if strategy in ['relevance', 'diversity', 'consistency', 'balanced']:
                        info['optimization_strategy'] = strategy
                    break
    elif 'custom' in basename:
        info['optimization_strategy'] = 'custom'
        # extract 'alpha' and 'beta'
        for part in parts:
            if part.startswith('a') and 'custom' not in part:
                try:
                    info['alpha'] = float(part[1:])
                except:
                    pass
            elif part.startswith('b') and 'custom' not in part:
                try:
                    info['beta'] = float(part[1:])
                except:
                    pass
    
    if 'fullperm' in basename:
        info['full_permutation'] = True
    
    return info


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_calls", type=int, default=5)
    parser.add_argument("--tops", type=int, default=1)
    parser.add_argument("--tails", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, 
                       choices=['19', '20', '21', '22', 'dev_small', 'nq_test'])
    parser.add_argument("--retriever", type=str, default='bm25', 
                       choices=['bm25', 'mt5', 'tct', 'oracle', 'reverse_oracle'])
    parser.add_argument("--optimization_strategy", type=str, default='balanced',
                       choices=['relevance', 'diversity', 'consistency', 'balanced', 'custom', 'none'])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--full_permutation", type=str, default='False', choices=['False', 'True'])
    parser.add_argument("--local_model_path", type=str, default="bert-large-uncased")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--auto_detect", action='store_true', 
                       help="Automatically detect and evaluate all generated files")
    parser.add_argument("--file_path", type=str, default=None,
                       help="Directly specify the file path to be evaluated")
    
    args = parser.parse_args()
    
    k = args.k
    num_calls = args.num_calls
    tops = args.tops
    tails = args.tails
    dataset_name = args.dataset_name
    retriever_name = args.retriever
    optimization_strategy = args.optimization_strategy
    alpha = args.alpha
    beta = args.beta
    full_permutation = eval(args.full_permutation)
    local_model_path = args.local_model_path
    cache_dir = args.cache_dir
    
    # Set cache directory
    if cache_dir:
        set_model_cache_dir(cache_dir)
    
    # Determine evaluation method
    eval_method = 'em/f1' if dataset_name == 'nq_test' else 'bertscore'
    
    print("==== Enhanced Evaluator ====")
    print(f"Dataset: {dataset_name}")
    print(f"Evaluation method: {eval_method}")
    
    # Prepare data
    if eval_method == 'bertscore':
        print(f"Using model: {local_model_path}")
        answer_scorer = load("bertscore", model_type=local_model_path)
        qids, qrels, doc_dict = prepare_qids_qrels_docdict(dataset_name)
    else:
        qids, gold_answers = prepare_qids_goldanswers(dataset_name)
    
    # Determine files to evaluate
    files_to_evaluate = []
    
    if args.file_path:
        # Directly specified file
        if os.path.exists(args.file_path):
            files_to_evaluate.append(args.file_path)
        else:
            print(f"Specified file does not exist: {args.file_path}")
            exit(1)
    elif args.auto_detect:
        # Automatically detect all relevant files
        pattern = f'./gen_results/*_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_*.json'
        files_to_evaluate = glob.glob(pattern)
        if not files_to_evaluate:
            print("No matching files found")
            exit(1)
        print(f"Found {len(files_to_evaluate)} files to evaluate:")
        for f in files_to_evaluate:
            print(f"  - {os.path.basename(f)}")
    else:
        # Find file based on parameters
        file_path = find_generation_file(
            k, num_calls, tops, tails, retriever_name, dataset_name,
            optimization_strategy, alpha, beta, full_permutation
        )
        if file_path:
            files_to_evaluate.append(file_path)
        else:
            print("No matching generation file found")
            print("Please check parameters or use --auto_detect option")
            exit(1)
    
    # Evaluate each file
    for file_path in files_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating file: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # Auto-detect file type
        file_info = auto_detect_file_type(file_path)
        print(f"Detected optimization strategy: {file_info['optimization_strategy']}")
        if file_info['alpha'] is not None:
            print(f"Alpha: {file_info['alpha']}, Beta: {file_info['beta']}")
        
        # Build evaluation result file path
        eval_file_path = file_path.replace('./gen_results/', './eval_results/').replace('.json', '_eval.json')
        
        # Ensure evaluation results directory exists
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        
        # Read generated answers
        try:
            with open(file_path, "r") as f:
                answer_book = json.load(f)
        except Exception as e:
            print(f"Failed to read answer file: {e}")
            continue
        
        # Create or read existing evaluation results
        try:
            with open(eval_file_path, "r") as f:
                existed_results = json.load(f)
                existed_qids = len(existed_results)
        except:
            existed_results = {}
            existed_qids = 0
        
        print(f"Already evaluated queries: {existed_qids}")
        print("Starting evaluation...")
        
        # Evaluation loop
        total_qids = len([str(id) for id in qids])
        for qid_idx, qid in enumerate([str(id) for id in qids[existed_qids:]]):
            print(f'Evaluating query {existed_qids + qid_idx + 1}/{total_qids}: {qid}')
            
            if str(qid) not in answer_book:
                print(f"  Warning: Query {qid} not found in answer file")
                continue
            
            eval_result_qid = {}
            
            # Iterate through each context strategy
            for strategy_name in answer_book[str(qid)].keys():
                print(f'  Strategy: {strategy_name}')
                eval_result_strategy = {}
                
                # Iterate through each call's results
                for call_idx in answer_book[str(qid)][strategy_name].keys():
                    to_eval = answer_book[str(qid)][strategy_name][call_idx]['answer']
                    
                    if eval_method == 'bertscore':
                        # BERTScore evaluation
                        docno_dict = get_docnos(qid=qid, doc_dict=doc_dict, qrels=qrels)
                        if dataset_name == 'dev_small':
                            r = eval_by_qrels(to_eval=to_eval, docno_dict=docno_dict, 
                                            qrel_levels=[1], local_model_path=local_model_path)
                        else:
                            r = eval_by_qrels(to_eval=to_eval, docno_dict=docno_dict, 
                                            local_model_path=local_model_path)
                    else:
                        # EM/F1 evaluation
                        gold_answers_qid = gold_answers[gold_answers.qid == int(qid)].gold_answer.values
                        r = eval_by_goldanswers(to_eval=to_eval, refs=gold_answers_qid)
                    
                    eval_result_strategy.update({call_idx: r})
                
                eval_result_qid.update({strategy_name: eval_result_strategy})
            
            # Update and save results
            existed_results.update({qid: eval_result_qid})
            
            with open(eval_file_path, "w+") as f:
                json.dump(existed_results, f, indent=4)
        
        print(f"=== File evaluation completed: {eval_file_path} ===")
    
    print(f"\n{'='*60}")
    print("All evaluations completed!")
    print(f"{'='*60}")