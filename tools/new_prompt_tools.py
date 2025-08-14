import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
from tools.permutation_generator import get_permutation

class RAGContextOptimizer:
    """
    RAG Context Optimizer using multi-criteria selection:
    - A: Relevance: Query-document similarity
    - B: Diversity: Inter-document diversity 
    - C: Consistency: Logical consistency between documents
    
    Optimization formula: alpha*A - beta*B + (1-alpha-beta)*C
    """
    
    def __init__(self, similarity_model=None, entailment_model=None):
        self.similarity_model = similarity_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.entailment_model = entailment_model or pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
    
    def compute_relevance(self, query: str, candidates: List[str]) -> np.ndarray:
        """
        A: Compute query-document relevance scores
        Returns: Array of relevance scores (higher is better)
        """
        if not candidates:
            return np.array([])
            
        query_embedding = self.similarity_model.encode([query])
        candidate_embeddings = self.similarity_model.encode(candidates)
        
        # Compute cosine similarity
        similarities = np.dot(query_embedding, candidate_embeddings.T).flatten()
        return similarities
    
    def compute_diversity(self, candidates: List[str]) -> np.ndarray:
        """
        B: Compute inter-document diversity scores
        Note: In the formula, diversity has a negative sign, so higher diversity actually reduces the total score
        Returns: Array of diversity scores (higher means more diverse, but will be subtracted in optimization)
        """
        if len(candidates) <= 1:
            return np.array([0.0] * len(candidates))  # Single document has zero diversity
        
        embeddings = self.similarity_model.encode(candidates)
        diversity_scores = []
        
        for i, embedding in enumerate(embeddings):
            # Compute average similarity with all other documents
            other_embeddings = np.concatenate([embeddings[:i], embeddings[i+1:]])
            if len(other_embeddings) > 0:
                similarities = np.dot(embedding.reshape(1, -1), other_embeddings.T).flatten()
                avg_similarity = np.mean(similarities)
                # Diversity = 1 - average similarity
                diversity = 1.0 - avg_similarity
                # Ensure diversity score is within reasonable bounds
                diversity = max(0.0, min(1.0, diversity))
            else:
                diversity = 0.0
            diversity_scores.append(diversity)
        
        return np.array(diversity_scores)
    
    def compute_consistency(self, candidates: List[str]) -> np.ndarray:
        """
        C: Compute logical consistency scores between documents
        Returns: Array of consistency scores (higher means better logical consistency)
        """
        if len(candidates) <= 1:
            return np.array([1.0] * len(candidates))  # Single document has maximum consistency
        
        consistency_scores = []
        for i, doc_i in enumerate(candidates):
            other_docs = candidates[:i] + candidates[i+1:]
            if not other_docs:
                consistency_scores.append(1.0)
                continue
            
            total_consistency = 0.0
            valid_pairs = 0
            
            for other_doc in other_docs:
                try:
                    # Use NLI model to determine entailment/contradiction relationships
                    result = self.entailment_model(
                        f"{doc_i} [SEP] {other_doc}",
                        candidate_labels=["entailment", "neutral", "contradiction"]
                    )
                    
                    # Compute consistency score: entailment=1.0, neutral=0.5, contradiction=0.0
                    consistency_score = 0.0
                    for label, score in zip(result['labels'], result['scores']):
                        if label == 'entailment':
                            consistency_score += score * 1.0
                        elif label == 'neutral':
                            consistency_score += score * 0.5
                        elif label == 'contradiction':
                            consistency_score += score * 0.0
                    
                    total_consistency += consistency_score
                    valid_pairs += 1
                except Exception as e:
                    print(f"Consistency calculation error: {e}")
                    # On error, assume neutral relationship
                    total_consistency += 0.5
                    valid_pairs += 1
            
            avg_consistency = total_consistency / valid_pairs if valid_pairs > 0 else 0.5
            consistency_scores.append(avg_consistency)
        
        return np.array(consistency_scores)
    
    def optimize_context_selection(self, query: str, documents: List[str], 
                                 k: int, alpha: float = 0.5, beta: float = 0.3) -> List[Tuple[str, float]]:
        """
        Optimize selection of top k documents using formula: alpha*A - beta*B + (1-alpha-beta)*C
        
        Args:
            query: Query text
            documents: List of candidate documents
            k: Number of documents to select
            alpha: Weight for relevance
            beta: Weight for diversity (note: negative in formula)
            
        Returns:
            Sorted documents with their composite scores
        """
        if not documents:
            return []
        
        if len(documents) <= k:
            # If fewer documents than k, return all
            return [(doc, 1.0) for doc in documents]
        
        # Validate weight parameters
        if alpha < 0 or beta < 0 or alpha + beta > 1:
            print(f"Warning: Invalid weight parameters alpha={alpha}, beta={beta}, using defaults")
            alpha, beta = 0.5, 0.3
        
        gamma = 1 - alpha - beta  # (1-alpha-beta)
        
        print(f"Optimization weights: alpha={alpha} (relevance), beta={beta} (diversity-negative), gamma={gamma} (consistency)")
        
        # Compute all scores
        relevance_scores = self.compute_relevance(query, documents)  # A
        diversity_scores = self.compute_diversity(documents)         # B
        consistency_scores = self.compute_consistency(documents)     # C
        
        # Normalize scores to [0,1] range
        def normalize_scores(scores):
            if len(scores) == 0:
                return scores
            min_score, max_score = scores.min(), scores.max()
            if max_score == min_score:
                return np.ones_like(scores)
            return (scores - min_score) / (max_score - min_score)
        
        relevance_norm = normalize_scores(relevance_scores)
        diversity_norm = normalize_scores(diversity_scores)
        consistency_norm = normalize_scores(consistency_scores)
        
        # Apply optimization formula: alpha*A - beta*B + (1-alpha-beta)*C
        composite_scores = (
            alpha * relevance_norm -
            beta * diversity_norm +
            gamma * consistency_norm
        )
        
        # Use greedy algorithm to further optimize diversity
        # This helps avoid selecting overly similar document combinations
        selected_docs = self._greedy_selection(
            documents, composite_scores, relevance_norm, k, 
            diversity_threshold=0.8
        )
        
        return selected_docs
    
    def _greedy_selection(self, documents: List[str], scores: np.ndarray, 
                         relevance_scores: np.ndarray, k: int, 
                         diversity_threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Greedy algorithm for document selection that avoids overly similar documents
        """
        if len(documents) == 0:
            return []
        
        selected_docs = []
        selected_indices = []
        available_indices = list(range(len(documents)))
        
        # First select the highest scoring document
        best_idx = np.argmax(scores)
        selected_docs.append((documents[best_idx], scores[best_idx]))
        selected_indices.append(best_idx)
        available_indices.remove(best_idx)
        
        # Select remaining documents one by one with diversity constraints
        for _ in range(k - 1):
            if not available_indices:
                break
            
            best_score = float('-inf')
            best_idx = -1
            
            for idx in available_indices:
                # Compute maximum similarity with already selected documents
                candidate_embedding = self.similarity_model.encode([documents[idx]])
                selected_embeddings = self.similarity_model.encode([doc for doc, _ in selected_docs])
                
                if len(selected_embeddings) > 0:
                    similarities = np.dot(candidate_embedding, selected_embeddings.T).flatten()
                    max_similarity = np.max(similarities)
                    
                    # Apply penalty if similarity exceeds threshold
                    diversity_penalty = 0.0
                    if max_similarity > diversity_threshold:
                        diversity_penalty = (max_similarity - diversity_threshold) * 2.0
                    
                    adjusted_score = scores[idx] - diversity_penalty
                else:
                    adjusted_score = scores[idx]
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_docs.append((documents[best_idx], scores[best_idx]))
                selected_indices.append(best_idx)
                available_indices.remove(best_idx)
        
        return selected_docs
    
    def rank_by_scores(self, candidates: List[str], scores: np.ndarray) -> List[Tuple[str, float]]:
        """Rank candidate documents by their scores"""
        ranked_pairs = list(zip(candidates, scores))
        ranked_pairs.sort(key=lambda x: x[1], reverse=True)
        return ranked_pairs


class PromptTools:
    """
    prompt tools class with integrated context optimization capabilities
    """
    
    def __init__(self, optimizer: RAGContextOptimizer = None):
        self.optimizer = optimizer or RAGContextOptimizer()
    
    def prepare_data(self, dataset_name: str, retriever_name='bm25'):
        """data preparation"""
        import pickle
        
        if dataset_name in ['19', '20', 'dev_small']:
            with open('./doc_dicts/msmarco_passage_dict.pkl', 'rb') as f:
                doc_dict = pickle.load(f)
        elif dataset_name in ['21', '22']:
            with open('./doc_dicts/msmarco_passage_v2_dict.pkl', 'rb') as f:
                doc_dict = pickle.load(f)
        elif dataset_name == 'nq_test':
            with open('./doc_dicts/nq_wiki_dict.pkl', 'rb') as f:
                doc_dict = pickle.load(f)
        elif dataset_name == 'hotpotqa_dev':
            with open('./doc_dicts/hotpotqa_wiki_dict.pkl', 'rb') as f:
                doc_dict = pickle.load(f)
        else:
            raise ValueError(f'Dataset {dataset_name} is not supported')
        
        queries = pd.read_csv(f'./queries/queries_{dataset_name}.csv')
        
        if dataset_name in ['dev_small', 'nq_test', 'hotpotqa_dev']:
            res = pd.read_csv(f'./res/{retriever_name}_{dataset_name}.csv')
        else:
            res = pd.read_csv(f'./res/{retriever_name}_dl_{dataset_name}.csv')
        
        return doc_dict, queries, res
    
    def compose_optimized_context(self, query: str, res: pd.DataFrame, qid: str, 
                                k: int, doc_dict: Dict[str, str],
                                optimization_strategy: str = 'balanced',
                                max_candidates: int = 20,
                                alpha: float = None, beta: float = None) -> Tuple[List[str], List[str]]:
        """
        Select context documents using optimization strategy
        
        Args:
            query: Query text
            res: Retrieval results DataFrame
            qid: Query ID
            k: Number of documents to select
            doc_dict: Document dictionary
            optimization_strategy: Optimization strategy ['relevance', 'diversity', 'consistency', 'balanced', 'custom']
            max_candidates: Maximum number of candidate documents to consider from retrieval results
            alpha: Relevance weight (used when strategy='custom')
            beta: Diversity weight (used when strategy='custom')
        
        Returns:
            (start_records, context_book): Starting rank records and context document list
        """
        res.qid = res.qid.astype('str')
        retrieved_for_q = res[res.qid == str(qid)]
        
        # Get candidate documents
        top_docnos = retrieved_for_q.head(max_candidates)['docno'].tolist()
        candidate_texts = [doc_dict[str(docno)] for docno in top_docnos if str(docno) in doc_dict]
        
        if len(candidate_texts) < k:
            print(f"Warning: Available documents ({len(candidate_texts)}) fewer than requested ({k})")
            k = len(candidate_texts)
        
        # Set alpha and beta parameters based on strategy
        if optimization_strategy == 'custom' and alpha is not None and beta is not None:
            # Use custom weights
            opt_alpha, opt_beta = alpha, beta
        else:
            # Predefined strategy weight mapping
            strategy_params = {
                'relevance': {'alpha': 1.0, 'beta': 0.0},      # Only consider relevance
                'diversity': {'alpha': 0.2, 'beta': 0.8},      # High diversity (low similarity)
                'consistency': {'alpha': 0.2, 'beta': 0.0},    # High consistency (alpha=0.2, beta=0.0 -> gamma=0.8)
                'balanced': {'alpha': 0.5, 'beta': 0.3}        # Balanced strategy (gamma=0.2)
            }
            
            params = strategy_params.get(optimization_strategy, strategy_params['balanced'])
            opt_alpha, opt_beta = params['alpha'], params['beta']
        
        print(f"Using optimization parameters: alpha={opt_alpha}, beta={opt_beta}, gamma={1-opt_alpha-opt_beta}")
        
        # Optimize document selection
        optimized_docs = self.optimizer.optimize_context_selection(
            query, candidate_texts, k, opt_alpha, opt_beta
        )
        
        # Build context
        context = ''
        for i, (doc_text, score) in enumerate(optimized_docs, 1):
            context += f'Context {i}: "{doc_text}";\n'
        
        strategy_name = f"optimized_{optimization_strategy}_a{opt_alpha}_b{opt_beta}"
        return [strategy_name], [context]
    
    def compose_context_with_optimization_and_permutations(
        self, query: str, res: pd.DataFrame, qid: str, k: int, doc_dict: Dict[str, str],
        optimization_strategy: str = 'balanced', full_permutations: bool = False,
        max_candidates: int = 20, alpha: float = None, beta: float = None
    ) -> Tuple[List[str], List[str]]:
        """
        Combine optimized selection with permutations for context composition
        """
        # First perform optimized selection
        _, optimized_contexts = self.compose_optimized_context(
            query, res, qid, k, doc_dict, optimization_strategy, max_candidates, alpha, beta
        )
        
        optimized_context = optimized_contexts[0]
        
        # Extract optimized document texts
        import re
        doc_texts = re.findall(r'Context \d+: "(.*?)";', optimized_context)
        
        if not full_permutations:
            # Use simplified permutation strategy
            permutation_book = get_permutation(doc_texts, len(doc_texts), full_permutations=False)
        else:
            # Use full permutations
            permutation_book = get_permutation(doc_texts, len(doc_texts), full_permutations=True)
        
        p_name_list = []
        context_book = []
        
        for p_name, p_batch_texts in permutation_book.items():
            context = ''
            for i, text in enumerate(p_batch_texts, 1):
                context += f'Context {i}: "{text}";\n'
            
            strategy_name = f'{optimization_strategy}_a{alpha or "def"}_b{beta or "def"}>{p_name}'
            p_name_list.append(strategy_name)
            context_book.append(context)
        
        return p_name_list, context_book


def used_preamble(long_answer=True):
    """Maintain the original preamble logic"""
    if long_answer:
        return "You are an expert at answering questions based on your own knowledge and related context. Please answer this question based on the given context. End your answer with STOP."
    else:
        return "You are an expert at answering questions based on your own knowledge and related context. Please answer this question based on the given context within 5 words. You should put your answer inside <answer> and </answer>. "

def used_preamble_0(long_answer=True):
    """Maintain the original 0-shot preamble logic"""
    if long_answer:
        return "You are an expert at answering questions based on your own knowledge. Please answer this question. End your answer with STOP."
    else:
        return "You are an expert at answering questions based on your own knowledge. Please answer this question within 5 words. You should put your answer inside <answer> and </answer>. "

def prompt_assembler_0(query: str, long_answer=True):
    """Maintain the original 0-shot prompt assembly logic"""
    preamble = used_preamble_0(long_answer)
    if long_answer:
        return f'{preamble} \nQuestion: "{query}"\nNow start your answer. \nAnswer: '
    else:
        return f'{preamble} \nQuestion: "{query}"\nNow start your answer. \nAnswer: <answer>'

def prompt_assembler(context: str, query: str, long_answer=True):
    """Maintain the original k-shot prompt assembly logic"""
    preamble = used_preamble(long_answer)
    if long_answer:
        return f'{preamble} \n{context}Question: "{query}"\nNow start your answer. \nAnswer: '
    else:
        return f'{preamble} \n{context}Question: "{query}"\nNow start your answer. \nAnswer: <answer>'


# Example usage and test functions
def test_optimizer():
    """Test optimizer functionality"""
    optimizer = RAGContextOptimizer()
    
    # Sample data
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Python is a popular programming language used in many applications including web development and data science.",
        "Natural language processing helps computers understand and process human language effectively.",
        "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning based."
    ]
    
    print("=== Testing ABC Score Calculations ===")
    relevance_scores = optimizer.compute_relevance(query, documents)
    diversity_scores = optimizer.compute_diversity(documents)
    consistency_scores = optimizer.compute_consistency(documents)
    
    print("Relevance Scores (A):", relevance_scores)
    print("Diversity Scores (B):", diversity_scores)
    print("Consistency Scores (C):", consistency_scores)
    
    print("\n=== Testing Optimization Results with Different Alpha/Beta Combinations ===")
    test_params = [
        (0.7, 0.2),  # Focus on relevance
        (0.3, 0.6),  # Focus on diversity (negative)
        (0.2, 0.1),  # Focus on consistency (gamma=0.7)
        (0.5, 0.3),  # Balanced strategy
    ]
    
    for alpha, beta in test_params:
        print(f"\nalpha={alpha}, beta={beta}, gamma={1-alpha-beta}")
        optimized_docs = optimizer.optimize_context_selection(query, documents, k=3, alpha=alpha, beta=beta)
        for i, (doc, score) in enumerate(optimized_docs):
            print(f"  {i+1}. {score:.3f}: {doc[:50]}...")

def test_enhanced_prompt_tools():
    """Test enhanced prompt tools' custom weight functionality"""
    print("\n=== Testing Enhanced Prompt Tools ===")
    
    # Actual data files would be required to run this
    # enhanced_tools = EnhancedPromptTools()
    # Since we lack actual data files, we'll just demonstrate the interface
    print("Enhanced prompt tools support the following optimization strategies:")
    print("1. 'relevance' - Focus only on relevance")
    print("2. 'diversity' - Emphasize diversity") 
    print("3. 'consistency' - Emphasize consistency")
    print("4. 'balanced' - Balanced strategy")
    print("5. 'custom' - Custom alpha/beta parameters")

if __name__ == "__main__":
    test_optimizer()
    test_enhanced_prompt_tools()