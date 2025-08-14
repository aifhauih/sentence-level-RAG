import os
import json
import pickle
import pandas as pd
from tqdm import tqdm

def create_dict_from_local_jsonl(jsonl_path='/mnt/RAG/rag-utility/data/corpus.jsonl'):
    """Build MSMARCO document dictionary from local jsonl file (full corpus)"""

    doc_dict = {}

    print(f"Reading from {jsonl_path}...")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Processing documents")):
                item = json.loads(line)
                doc_id = str(item["docid"])
                doc_text = item.get("text", "")
                doc_dict[doc_id] = doc_text

                if (i + 1) % 100000 == 0:
                    print(f"Processed {i + 1} documents...")

        print(f"Created dictionary with {len(doc_dict)} documents")

        # Save dictionary
        output_path = './msmarco_passage_dict.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(doc_dict, f)

        print(f"Dictionary saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def create_selective_dict_from_local_jsonl(jsonl_path='/mnt/RAG/rag-utility/data/corpus.jsonl'):
    """Create dictionary only for needed documents (based on retrieval results)"""

    needed_docnos = set()
    res_files = [
        'res/bm25_nq_test.csv',
        'res/bm25_dev_small.csv',
        'res/bm25_dl_19.csv',
        'res/bm25_dl_20.csv',
        'res/e5_nq_test.csv',
        'res/mt5_nq_test.csv',
        # Add more retrieval result files as needed
    ]

    for file_path in res_files:
        try:
            df = pd.read_csv(file_path)
            file_docnos = set(str(docno) for docno in df['docno'].unique())
            needed_docnos.update(file_docnos)
            print(f"Added {len(file_docnos)} docnos from {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Total needed documents: {len(needed_docnos)}")

    doc_dict = {}
    found_count = 0
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Searching documents")):
                item = json.loads(line)
                doc_id = str(item["docid"])

                if doc_id in needed_docnos:
                    doc_dict[doc_id] = item["text"]
                    found_count += 1

                if found_count == len(needed_docnos):
                    break

        print(f"Found {found_count} / {len(needed_docnos)} documents")

        output_path = './msmarco_passage_dict.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(doc_dict, f)

        print(f"Selective dictionary saved to {output_path}")

        missing = needed_docnos - set(doc_dict.keys())
        if missing:
            print(f"Missing {len(missing)} documents. Sample: {list(missing)[:5]}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def verify_dict():
    """Verify if the created document dictionary is successful"""
    path = './doc_dicts/msmarco_passage_dict.pkl'
    if not os.path.exists(path):
        print("Dictionary file not found.")
        return False

    try:
        with open(path, 'rb') as f:
            doc_dict = pickle.load(f)

        print(f"Dictionary contains {len(doc_dict)} documents")

        # Check docnos in qrels.csv
        qrels = pd.read_csv('./qrels/qrels.csv')
        qrels['docno'] = qrels['docno'].astype('str')
        missing_docnos = set(qrels['docno']) - set(doc_dict.keys())
        if missing_docnos:
            print(f"The following {len(missing_docnos)} docnos are missing from doc_dict:")
            print(list(missing_docnos)[:10])
            if '1113693' in missing_docnos:
                print("docno '1113693' is missing from doc_dict")
        else:
            print("All qrels docnos are present in doc_dict")

        # Print samples
        sample_keys = list(doc_dict.keys())[:3]
        for key in sample_keys:
            text = doc_dict[key]
            print(f"Doc ID: {key}")
            print(f"Text: {text[:100]}...")
            print("-" * 50)

        return True
    except Exception as e:
        print(f"Error verifying dictionary: {e}")
        return False

if __name__ == "__main__":
    print("MSMARCO Dictionary Creator (From Local corpus.jsonl)")
    print("=" * 60)
    print("1. Create full dictionary (all documents)")
    print("2. Create selective dictionary (only needed documents) - RECOMMENDED")
    print("3. Verify existing dictionary")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print("Creating full dictionary from local corpus.jsonl...")
        success = create_dict_from_local_jsonl()
    elif choice == "2":
        print("Creating selective dictionary from local corpus.jsonl...")
        success = create_selective_dict_from_local_jsonl()
    elif choice == "3":
        print("Verifying dictionary...")
        success = verify_dict()
    else:
        print("Invalid choice!")
        success = False
    
    if success:
        print("Operation completed successfully!")
    else:
        print("Operation failed!")