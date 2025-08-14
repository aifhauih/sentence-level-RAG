from evaluate import load
import pandas as pd
import pickle
import json
import argparse
import os
from tools import eval_tools

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Set transformers cache directory (must be set before importing transformers)
def set_model_cache_dir(cache_dir):
    if cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        print(f"Setting model cache directory: {cache_dir}")

def prepare_qids_qrels_docdict(dataset_name):
    if dataset_name in ['21', '22']:
        with open('./doc_dicts/msmarco_passage_v2_dict.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
    else:
        with open('./doc_dicts/msmarco_passage_dict.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            
    print("Document dictionary loaded")
    
    queries = pd.read_csv(f'./queries/queries_{dataset_name}.csv')
    queries['qid'] = queries['qid'].astype('str')
    qids = queries.qid.tolist()
    if dataset_name in ['21', '22']:
        qrels = pd.read_csv('./qrels/qrels_v2.csv')
    elif dataset_name == 'dev_small':
        qrels = pd.read_csv('./qrels/qrels_dev.csv')
    else:
        qrels = pd.read_csv('./qrels/qrels.csv')
    
    print("Data loaded")
    qrels['qid'] = qrels['qid'].astype('str')
    qrels['docno'] = qrels['docno'].astype('str')
    
    return qids, qrels, doc_dict

def prepare_qids_goldanswers(dataset_name):
    queries = pd.read_csv(f'./queries/queries_{dataset_name}.csv')
    queries['qid'] = queries['qid'].astype('str')
    qids = queries.qid.tolist()
    nq_gds = pd.read_csv('./golden_answers/gdas_nq_test.csv')

    return qids, nq_gds

def get_docnos(qid, doc_dict, qrels):
    # Modified section: Skip missing docnos to avoid KeyError
    docnos_0, docnos_1, docnos_2, docnos_3 = [], [], [], []
    missing_docnos = []
    
    # Process label=0 documents
    for docno in qrels[(qrels.qid == qid) & (qrels.label == 0)].docno.tolist():
        if str(docno) in doc_dict:
            docnos_0.append(doc_dict[str(docno)])
        else:
            missing_docnos.append(str(docno))
    
    # Process label=1 documents
    for docno in qrels[(qrels.qid == qid) & (qrels.label == 1)].docno.tolist():
        if str(docno) in doc_dict:
            docnos_1.append(doc_dict[str(docno)])
        else:
            missing_docnos.append(str(docno))
    
    # Process label=2 documents
    for docno in qrels[(qrels.qid == qid) & (qrels.label == 2)].docno.tolist():
        if str(docno) in doc_dict:
            docnos_2.append(doc_dict[str(docno)])
        else:
            missing_docnos.append(str(docno))
    
    # Process label=3 documents
    for docno in qrels[(qrels.qid == qid) & (qrels.label == 3)].docno.tolist():
        if str(docno) in doc_dict:
            docnos_3.append(doc_dict[str(docno)])
        else:
            missing_docnos.append(str(docno))
    
    # Save missing docnos to file
    if missing_docnos:
        # with open(f'./missing_docnos_qid_{qid}.txt', 'w') as f:
        #     f.write('\n'.join(missing_docnos))
        # print(f"Missing docnos saved to ./missing_docnos_qid_{qid}.txt")
    
    docno_dict = {0: docnos_0, 1: docnos_1, 2: docnos_2, 3: docnos_3}
    return docno_dict

def evaluator(to_eval: str, docno_dict: dict, qrel_level: int, local_model_path: str = "bert-large-uncased"):
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
    
    # Use specified model path (BERTScore doesn't support cache_dir parameter)
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
    r = {}
    for level in qrel_levels:
        r.update({f'qrel_{level}': evaluator(to_eval, docno_dict, level, local_model_path)})
    return r

def eval_by_goldanswers(to_eval: str, refs):
    r = {}

    try:
        r.update({'EM': eval_tools.cal_em(to_eval, refs)})
        r.update({'F1': eval_tools.cal_f1(to_eval, refs)})
    except:
        r.update({'EM': -1})
        r.update({'F1': -1})        
    return r

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_calls", type=int, default=5)
    parser.add_argument("--tops", type=int, default=1)
    parser.add_argument("--tails", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, choices=['19', '20', '21', '22', 'dev_small', 'nq_test'])
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'mt5', 'tct', 'oracle', 'reverse_oracle'])
    parser.add_argument("--suffix", type=str, default='', choices=['', '_p'])
    # Add local model path parameter
    parser.add_argument("--local_model_path", type=str, default="bert-large-uncased", help="Path to local BERT model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for transformers models")
    args = parser.parse_args()

    k = args.k
    num_calls = args.num_calls
    # Start control parameters
    tops = args.tops
    tails = args.tails
    dataset_name = args.dataset_name
    retriever_name = args.retriever
    local_model_path = args.local_model_path
    cache_dir = args.cache_dir
    
    # Set cache directory (must be set before loading models)
    if cache_dir:
        set_model_cache_dir(cache_dir)
    
    if(dataset_name == 'nq_test'):
        eval_method = 'em/f1'
    else:
        eval_method = 'bertscore'
    if(k==0):
        tops, tails, retriever_name = 0, 0, 'bm25'
    suffix = args.suffix
    if(suffix == '_p'):
        print('Now evaluating permutation results!')

    [name_start, name_end] = ['short', 'concise'] if dataset_name=='nq_test' else ['random', 'prompt1']
    
    file_path = f'./gen_results/{name_start}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}{suffix}_{name_end}.json'
    eval_file_path = f'./eval_results/{name_start}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}{suffix}_{name_end}_eval.json'

    # Experiment begins
    print("==== Begin Evaluation ====")
    # Prepare data
    if(eval_method == 'bertscore'):
        print(f"Using model: {local_model_path}")
        answer_scorer = load("bertscore", model_type=local_model_path)
        qids, qrels, doc_dict = prepare_qids_qrels_docdict(dataset_name)
    elif(eval_method == 'em/f1'):
        qids, gold_answers = prepare_qids_goldanswers(dataset_name)
    else:
        print(f'This evaluation method is not currently supported')
    
    # Read the generated answers
    try:
        with open(file=file_path, mode="r") as f:
            answer_book = json.load(f)
    except:
        print(f'Answer book {file_path} does not exist.')
        answer_book = {}
    
    # Create the evaluation file
    try:
        with open(file=eval_file_path, mode="r") as f:
            existed_results = json.load(f)
            existed_qids = len(existed_results)
    except:
        existed_results = {}
        existed_qids = 0
    print("Answer data loaded")
    
    for qid in [str(id) for id in qids[existed_qids:]]:
        print(f'Evaluating Qid={qid}')
        eval_result_qid = {}
        for start in answer_book[str(qid)].keys():
            print(f'Start={start}, batch={k}')
            eval_result_start = {}
            for i in answer_book[str(qid)][str(start)].keys():
                print(f'\nCall {i}')
                to_eval = answer_book[str(qid)][str(start)][str(i)]['answer']
                
                if(dataset_name == 'dev_small'):
                    docno_dict = get_docnos(qid=qid, doc_dict=doc_dict, qrels=qrels)
                    r = eval_by_qrels(to_eval=to_eval, docno_dict=docno_dict, qrel_levels=[1], local_model_path=local_model_path)
                elif(dataset_name in ['19', '20', '21', '22']):
                    docno_dict = get_docnos(qid=qid, doc_dict=doc_dict, qrels=qrels)
                    r = eval_by_qrels(to_eval=to_eval, docno_dict=docno_dict, local_model_path=local_model_path)
                elif(dataset_name == 'nq_test'):
                    gold_answers_qid = gold_answers[gold_answers.qid==int(qid)].gold_answer.values
                    r = eval_by_goldanswers(to_eval=to_eval, refs=gold_answers_qid)
                else:
                    print('This dataset doesn\'t have reference answers')
                eval_result_start.update({i: r})
            eval_result_qid.update({start: eval_result_start})
        
        with open(file=eval_file_path, mode="r") as f:
            existed_results.update({qid: eval_result_qid})

        with open(file=eval_file_path, mode="w+") as f:
            json.dump(existed_results, f, indent=4)