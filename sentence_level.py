import numpy as np
import pandas as pd
import json
import pickle
import re
from typing import List, Tuple, Dict, Any
import jieba
import argparse
import os
import logging
from datetime import datetime
from tools import llama_tools, experiment_tools
from tools.new_prompt_tools import prompt_assembler, RAGContextOptimizer
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentence_level_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SentenceLevelRAG")

# Custom sentence segmentation function based on punctuation
def custom_sent_tokenize(text: str) -> List[str]:
    """Sentence segmentation using punctuation, works for both Chinese and English"""
    sentence_enders = re.compile(r'[.!?。！？]')
    sentences = []
    current_sentence = []
    for char in text:
        current_sentence.append(char)
        if sentence_enders.match(char):
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
            current_sentence = []
    if current_sentence:
        sentence = ''.join(current_sentence).strip()
        if sentence:
            sentences.append(sentence)
    return sentences

class SentenceLevelRAGSystem:
    """Sentence-level RAG system with enhanced context optimization"""
    
    def __init__(self, optimizer: RAGContextOptimizer = None, similarity_model=None):
        self.optimizer = optimizer or RAGContextOptimizer()
        self.similarity_model = similarity_model or SentenceTransformer('all-MiniLM-L6-v2')
    
    def prepare_sentence_level_data(self, dataset_name: str, retriever_name: str = 'bm25'):
        """Prepare sentence-level data"""
        try:
            from tools.enhanced_prompt_tools import EnhancedPromptTools
            enhanced_tools = EnhancedPromptTools()
            doc_dict, queries, res = enhanced_tools.prepare_data(dataset_name, retriever_name)
            return doc_dict, queries, res
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            raise
    
    def extract_sentences_from_document(self, document: str, 
                                      min_length: int = 20, 
                                      max_length: int = 300) -> List[str]:
        """Extract sentences from a single document"""
        sentences = []
        try:
            raw_sentences = custom_sent_tokenize(document)
            for sentence in raw_sentences:
                if min_length <= len(sentence.strip()) <= max_length:
                    clean_sentence = ' '.join(jieba.cut(sentence, cut_all=False)).strip()
                    clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
                    if clean_sentence and not clean_sentence.endswith('.'):
                        clean_sentence += '.'
                    if clean_sentence:
                        sentences.append(clean_sentence)
        except Exception as e:
            logger.error(f"Document processing error: {e}")
        return sentences
    
    def retrieve_and_rank_sentences(self, query: str, sentences: List[str], k: int) -> List[str]:
        """Retrieve and rank sentences, return top k most relevant sentences"""
        if not sentences:
            return []
        
        query_embedding = self.similarity_model.encode([query])
        sentence_embeddings = self.similarity_model.encode(sentences)
        similarities = np.dot(query_embedding, sentence_embeddings.T).flatten()
        
        # Get sorted indices
        ranked_indices = np.argsort(similarities)[::-1][:k]
        ranked_sentences = [sentences[i] for i in ranked_indices]
        return ranked_sentences
    
    def compose_sentence_level_context(self, query: str, res: pd.DataFrame, qid: str,
                                     k: int, doc_dict: Dict[str, str],
                                     top_docs: int = 10,
                                     optimization_strategy: str = 'balanced',
                                     alpha: float = None, beta: float = None) -> Tuple[List[str], List[str]]:
        """
        Assemble sentence-level context by first retrieving relevant documents, 
        extracting sentences, directly retrieving most relevant sentences, 
        and finally optimizing selection
        
        Args:
            query: Query text
            res: Retrieval results DataFrame
            qid: Query ID
            k: Number of sentences to select
            doc_dict: Document dictionary
            top_docs: Number of top documents to extract sentences from
            optimization_strategy: Optimization strategy
            alpha: Relevance weight
            beta: Diversity weight
        
        Returns:
            (strategy_names, contexts)
        """
        try:
            res.qid = res.qid.astype('str')
            retrieved_for_q = res[res.qid == str(qid)]
            
            top_docnos = retrieved_for_q.head(top_docs)['docno'].tolist()
            top_documents = [doc_dict.get(str(docno), "") for docno in top_docnos]
            top_documents = [doc for doc in top_documents if doc]
            
            if not top_documents:
                logger.warning(f"No documents available for query {qid}")
                return ["no_docs"], [""]
            
            # Extract sentences from each document and directly retrieve most relevant sentences
            all_candidate_sentences = []
            for doc in top_documents:
                sentences = self.extract_sentences_from_document(doc)
                if sentences:
                    all_candidate_sentences.extend(sentences)
            
            if not all_candidate_sentences:
                logger.warning(f"No sentences extracted for query {qid}")
                return ["no_sentences"], [""]
            
            logger.info(f"Extracted {len(all_candidate_sentences)} sentences from {len(top_documents)} documents, Query ID: {qid}")
            
            # Directly retrieve top k most relevant sentences
            initial_sentences = self.retrieve_and_rank_sentences(query, all_candidate_sentences, k)
            if not initial_sentences:
                logger.warning(f"No relevant sentences retrieved for query {qid}")
                return ["no_retrieved_sentences"], [""]
            
            logger.info(f"Retrieved {len(initial_sentences)} initial relevant sentences")
            
            # Set optimization parameters
            if optimization_strategy == 'custom' and alpha is not None and beta is not None:
                opt_alpha, opt_beta = alpha, beta
            else:
                strategy_params = {
                    'relevance': {'alpha': 1.0, 'beta': 0.0},
                    'diversity': {'alpha': 0.2, 'beta': 0.8},
                    'consistency': {'alpha': 0.2, 'beta': 0.0},
                    'balanced': {'alpha': 0.5, 'beta': 0.3}
                }
                params = strategy_params.get(optimization_strategy, strategy_params['balanced'])
                opt_alpha, opt_beta = params['alpha'], params['beta']
            
            # Optimize sentence selection
            selected_sentences = self.optimizer.optimize_context_selection(
                query=query,
                documents=initial_sentences,
                k=k,
                alpha=opt_alpha,
                beta=opt_beta
            )
            
            if not selected_sentences:
                logger.warning(f"No sentences selected for query {qid}")
                return ["no_selection"], [""]
            
            # Build context
            context = ""
            for i, (sentence, score) in enumerate(selected_sentences, 1):
                context += f"Context {i}: \"{sentence}\"\n"
            
            strategy_name = f"sentence_level_k{k}_top{top_docs}_{optimization_strategy}_a{opt_alpha}_b{opt_beta}"
            return [strategy_name], [context]
        except Exception as e:
            logger.error(f"Context assembly error for Query ID {qid}: {e}")
            return ["error"], [""]
    
    def run_sentence_level_experiment(self, dataset_name: str, retriever_name: str,
                                    k: int, num_calls: int, temperature: float,
                                    long_answer: bool = True, top_docs: int = 10,
                                    optimization_strategy: str = 'balanced',
                                    alpha: float = None, beta: float = None):
        """Run sentence-level experiment, saving results after each response generation"""
        try:
            doc_dict, queries, res = self.prepare_sentence_level_data(dataset_name, retriever_name)
            
            llm = llama_tools.load_llama()
            
            short_answer_identifier = 'random' if long_answer else 'short'
            file_name = f'./gen_results/{short_answer_identifier}_answers_{k}shot_{num_calls}calls_0_0_{retriever_name}_dl_{dataset_name}_sentence_level_opt_{optimization_strategy}.json'
            
            try:
                with open(file_name, "r") as f:
                    result_to_write = json.load(f)
                    existed_qids = list(result_to_write.keys())
            except:
                result_to_write = {}
                existed_qids = []
            
            logger.info(f"Starting sentence-level experiment. Dataset: {dataset_name}, Retriever: {retriever_name}")
            logger.info(f"Existing query IDs count: {len(existed_qids)}")
            
            for idx, (qid, query) in enumerate(zip(queries['qid'].tolist(), queries['query'].tolist())):
                qid = str(qid)
                if qid in existed_qids:
                    logger.info(f"Skipping already processed query ID {idx+1}/{len(queries)}: {qid}")
                    continue
                
                logger.info(f"Processing query ID {idx+1}/{len(queries)}: {qid} - {query[:50]}...")
                
                strategy_names, contexts = self.compose_sentence_level_context(
                    query=query, res=res, qid=qid, k=k, doc_dict=doc_dict,
                    top_docs=top_docs, optimization_strategy=optimization_strategy,
                    alpha=alpha, beta=beta
                )
                
                varying_context_result = {}
                
                for strategy_name, context in zip(strategy_names, contexts):
                    try:
                        llm.set_seed(1000)
                        logger.info(f"\tStrategy: {strategy_name}, Context length: {len(context)} chars")
                        
                        prompt = prompt_assembler(context, query, long_answer)
                        
                        multi_call_results = {}
                        for j in range(num_calls):
                            logger.info(f"\t\tCall {j+1}/{num_calls}")
                            result = llama_tools.single_call(
                                llm=llm, prompt=prompt, temperature=temperature, long_answer=long_answer
                            )
                            multi_call_results[str(j)] = result
                        
                        varying_context_result[strategy_name] = multi_call_results
                    except Exception as e:
                        logger.error(f"Error processing strategy {strategy_name} for Query ID {qid}: {e}")
                
                result_to_write[qid] = varying_context_result
                
                try:
                    experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)
                    logger.info(f"Saved results for Query ID {qid} to {file_name}")
                except Exception as e:
                    logger.error(f"Failed to save results for Query ID {qid}: {e}")
            
            logger.info(f"Experiment completed. Results saved to: {file_name}")
            return file_name
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            try:
                del llm
            except:
                pass

class SentenceLevelEvaluator:
    """Sentence-level evaluator"""
    
    def __init__(self, local_model_path: str = "bert-large-uncased"):
        try:
            from evaluate import load
            self.answer_scorer = load("bertscore", model_type=local_model_path)
            self.local_model_path = local_model_path
        except ImportError:
            logger.error("Evaluation library not found. Please install evaluate.")
            raise
    
    def evaluate_sentence_level_results(self, gen_file_path: str, dataset_name: str):
        """Evaluate sentence-level generation results"""
        try:
            if dataset_name == 'nq_test':
                gold_answers = pd.read_csv('./golden_answers/gdas_nq_test.csv')
                eval_method = 'em_f1'
            else:
                qids, qrels, doc_dict = self._prepare_qrels_data(dataset_name)
                eval_method = 'bertscore'
            
            with open(gen_file_path, 'r') as f:
                answer_book = json.load(f)
            
            eval_file_path = gen_file_path.replace('gen_results', 'eval_results').replace('.json', '_eval.json')
            os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
            
            eval_results = {}
            total_qids = len(answer_book)
            
            for idx, qid in enumerate(answer_book.keys()):
                logger.info(f"Evaluating query ID {idx+1}/{total_qids}: {qid}")
                eval_result_qid = {}
                
                for strategy_name in answer_book[qid].keys():
                    logger.info(f"\tStrategy: {strategy_name}")
                    eval_result_strategy = {}
                    
                    for call_idx, call_result in answer_book[qid][strategy_name].items():
                        to_eval = call_result['answer']
                        
                        if eval_method == 'em_f1':
                            try:
                                from tools.eval_tools import cal_em, cal_f1
                                gold_answers_qid = gold_answers[gold_answers.qid == int(qid)].gold_answer.values
                                r = {
                                    'EM': cal_em(to_eval, gold_answers_qid),
                                    'F1': cal_f1(to_eval, gold_answers_qid)
                                }
                            except Exception as e:
                                logger.error(f"EM/F1 evaluation error for Query ID {qid}: {e}")
                                r = {'EM': -1, 'F1': -1}
                        else:
                            try:
                                docno_dict = self._get_docnos(qid, doc_dict, qrels)
                                r = self._eval_by_qrels(to_eval, docno_dict)
                            except Exception as e:
                                logger.error(f"BERTScore evaluation error for Query ID {qid}: {e}")
                                r = {
                                    'qrel_2': {'precision': {'avg': -1}, 'recall': {'avg': -1}, 'f1': {'avg': -1}},
                                    'qrel_3': {'precision': {'avg': -1}, 'recall': {'avg': -1}, 'f1': {'avg': -1}}
                                }
                        
                        eval_result_strategy[call_idx] = r
                    
                    eval_result_qid[strategy_name] = eval_result_strategy
                
                eval_results[qid] = eval_result_qid
                
                try:
                    with open(eval_file_path, 'w') as f:
                        json.dump(eval_results, f, indent=4)
                    logger.info(f"Saved evaluation results for Query ID {qid} to {eval_file_path}")
                except Exception as e:
                    logger.error(f"Failed to save evaluation results for Query ID {qid}: {e}")
            
            logger.info(f"Sentence-level evaluation completed. Results saved to: {eval_file_path}")
            return eval_file_path
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _prepare_qrels_data(self, dataset_name: str):
        """Prepare qrels data"""
        try:
            if dataset_name in ['21', '22']:
                with open('./doc_dicts/msmarco_passage_v2_dict.pkl', 'rb') as f:
                    doc_dict = pickle.load(f)
                qrels = pd.read_csv('./qrels/qrels_v2.csv')
            elif dataset_name == 'dev_small':
                with open('./doc_dicts/msmarco_passage_dict.pkl', 'rb') as f:
                    doc_dict = pickle.load(f)
                qrels = pd.read_csv('./qrels/qrels_dev.csv')
            else:
                with open('./doc_dicts/msmarco_passage_dict.pkl', 'rb') as f:
                    doc_dict = pickle.load(f)
                qrels = pd.read_csv('./qrels/qrels.csv')
            
            queries = pd.read_csv(f'./queries/queries_{dataset_name}.csv')
            qids = queries.qid.astype('str').tolist()
            qrels['qid'] = qrels['qid'].astype('str')
            qrels['docno'] = qrels['docno'].astype('str')
            
            return qids, qrels, doc_dict
        except Exception as e:
            logger.error(f"Error preparing qrels data: {e}")
            raise
    
    def _get_docnos(self, qid: str, doc_dict: Dict, qrels: pd.DataFrame):
        """Get relevant documents"""
        try:
            docnos_dict = {0: [], 1: [], 2: [], 3: []}
            
            for label in [0, 1, 2, 3]:
                docnos = qrels[(qrels.qid == qid) & (qrels.label == label)].docno.tolist()
                texts = [doc_dict.get(str(docno), "") for docno in docnos]
                texts = [text for text in texts if text]
                docnos_dict[label] = texts
            
            return docnos_dict
        except Exception as e:
            logger.error(f"Error getting documents for Query ID {qid}: {e}")
            return {0: [], 1: [], 2: [], 3: []}
    
    def _eval_by_qrels(self, to_eval: str, docno_dict: Dict, qrel_levels: List[int] = [2, 3]):
        """Evaluate using BERTScore with qrels"""
        r = {}
        for level in qrel_levels:
            doc_texts = docno_dict.get(level, [])
            if not doc_texts:
                r[f'qrel_{level}'] = {
                    'precision': {'avg': -1, 'max': -1},
                    'recall': {'avg': -1, 'max': -1},
                    'f1': {'avg': -1, 'max': -1}
                }
                continue
            
            try:
                predictions = [to_eval] * len(doc_texts)
                
                results = self.answer_scorer.compute(
                    predictions=predictions,
                    references=doc_texts,
                    lang="en",
                    model_type=self.local_model_path,
                    verbose=False
                )
                
                precisions, recalls, f1s = results['precision'], results['recall'], results['f1']
                
                r[f'qrel_{level}'] = {
                    'precision': {'avg': sum(precisions)/len(precisions), 'max': max(precisions)},
                    'recall': {'avg': sum(recalls)/len(recalls), 'max': max(recalls)},
                    'f1': {'avg': sum(f1s)/len(f1s), 'max': max(f1s)}
                }
            except Exception as e:
                logger.error(f"BERTScore calculation error for level {level}: {e}")
                r[f'qrel_{level}'] = {
                    'precision': {'avg': -1, 'max': -1},
                    'recall': {'avg': -1, 'max': -1},
                    'f1': {'avg': -1, 'max': -1}
                }
        
        return r

def main():
    parser = argparse.ArgumentParser(description='Sentence-level K-Shot RAG Experiment')
    parser.add_argument('--mode', type=str, choices=['run', 'evaluate'], default='run', 
                        help='Operation mode: run experiment or evaluate results')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['19', '20', '21', '22', 'dev_small', 'nq_test'],
                        help='Dataset name')
    parser.add_argument('--retriever', type=str, default='bm25',
                        choices=['bm25', 'mt5', 'tct', 'oracle', 'reverse_oracle'],
                        help='Retriever name')
    parser.add_argument('--k', type=int, default=3, help='Number of sentences to select')
    parser.add_argument('--num_calls', type=int, default=3, 
                        help='Number of LLM calls per query')
    parser.add_argument('--top_docs', type=int, default=10, 
                        help='Number of top documents to extract sentences from')
    parser.add_argument('--gen_file', type=str, 
                        help='Path to generated results file in evaluate mode')
    parser.add_argument('--optimization_strategy', type=str, default='balanced',
                        choices=['relevance', 'diversity', 'consistency', 'balanced', 'custom'],
                        help='Optimization strategy')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Relevance weight (only used with custom strategy)')
    parser.add_argument('--beta', type=float, default=None,
                        help='Diversity weight (only used with custom strategy)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting sentence-level experiment with parameters: {vars(args)}")
    
    try:
        if args.mode == 'run':
            system = SentenceLevelRAGSystem()
            gen_file = system.run_sentence_level_experiment(
                dataset_name=args.dataset_name,
                retriever_name=args.retriever,
                k=args.k,
                num_calls=args.num_calls,
                temperature=0.3,
                long_answer=True,
                top_docs=args.top_docs,
                optimization_strategy=args.optimization_strategy,
                alpha=args.alpha,
                beta=args.beta
            )
            logger.info(f"Experiment completed. Generated file: {gen_file}")
            
            evaluator = SentenceLevelEvaluator()
            eval_file = evaluator.evaluate_sentence_level_results(gen_file, args.dataset_name)
            logger.info(f"Evaluation completed. Results file: {eval_file}")
            
        elif args.mode == 'evaluate' and args.gen_file:
            evaluator = SentenceLevelEvaluator()
            eval_file = evaluator.evaluate_sentence_level_results(args.gen_file, args.dataset_name)
            logger.info(f"Evaluation completed. Results file: {eval_file}")
        else:
            logger.error("Invalid arguments. In evaluate mode, please provide --gen_file")
    except Exception as e:
        logger.exception("Critical error occurred during operation")
        raise

if __name__ == "__main__":
    main()