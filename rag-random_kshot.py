from tools import llama_tools, prompt_tools, experiment_tools
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
    parser.add_argument("--dataset_name", type=str, choices=['19', '20', '21', '22', 'dev_small', 'nq_test'])
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'mt5', 'tct', 'oracle', 'reverse_oracle'])
    parser.add_argument("--long_answer", type=str, default='True', choices=['False', 'True'])
    # parser.add_argument("--full_permutation", type=str, default='False', choices=['False', 'True'])
    args = parser.parse_args()

    k = args.k
    step = args.step
    num_calls = args.num_calls
    # start control parameters
    tops = args.tops
    tails = args.tails
    temperature = args.temperature
    dataset_name = args.dataset_name
    retriever_name = args.retriever
    long_answer = True if args.long_answer == 'True' else False
    short_answer_identifier = 'random' if long_answer else 'short'
    
    full_permutation = False
    print("Using original context order only (no permutations).")
    
    # load the llm
    llm = llama_tools.load_llama()
    # load needed data
    doc_dict, queries, res = prompt_tools.prepare_data(dataset_name, retriever_name)
    
    setting_file_name = f'./gen_results/{short_answer_identifier}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_settings_no_permutation.json'
    setting_record = {'k': k, 'full_permutation': full_permutation, 'num_calls': num_calls, 'step': step, 'tops': tops, 'tails': tails,
                      'temperature': temperature, 'query_set': dataset_name, 'retriever': retriever_name, 'long_answer?': long_answer}
    setting_record.update({'--experiment_start_at': str(datetime.datetime.now())})
    
    with open(setting_file_name, "w", encoding='UTF-8') as f:
        json.dump(setting_record, f, indent=4)

    file_name = f'./gen_results/{short_answer_identifier}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_no_permutation.json'

    try:
        with open(file_name, "r") as f:
            result_to_write = json.load(f)
            existed_qids_list = list(result_to_write.keys())
    except:
        result_to_write = {}
        existed_qids_list = []

    q_no = 0
    for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
        print(f'q_number={q_no}--{qid}')
        if str(qid) in existed_qids_list:
            print("Already generated, next!")
            q_no += 1
            continue
        
        q_no += 1
        
        start_records, context_book = prompt_tools.compose_context_with_permutations(
            qid=qid, res=res, k=k, step=step,
            tops=tops, tails=tails, doc_dict=doc_dict, 
            full_permutations=full_permutation
        )
        print('start records: ', start_records)

        if len(context_book) > 0:
            start = start_records[0]
            context = context_book[0]
            
            llm.set_seed(1000)
            print(f'\tstart_rank.{start}')
            print("==>", context)
            prompt = prompt_tools.prompt_assembler(context, query, long_answer)
            print(prompt)
            
            multi_call_results = {}
            
            # Save problem and context info
            qid_result = {
                "problem": query,
                "context": context,
                "responses": {}
            }
            
            # Add error handling for LLM calls
            for j in range(num_calls):
                print(f'\t\tno.{j}')
                try:
                    result = llama_tools.single_call(llm=llm, prompt=prompt, temperature=temperature)
                    print("====>", result)
                    qid_result["responses"][j] = result
                    
                    # Save after each response
                    result_to_write[qid] = qid_result
                    try:
                        experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)
                    except Exception as save_error:
                        print(f"Error saving after response {j}: {save_error}")
                        
                except Exception as e:
                    print(f"Error in call {j}: {e}")
                    qid_result["responses"][j] = f"Error: {str(e)}"
                    
                    # Save error result too
                    result_to_write[qid] = qid_result
                    try:
                        experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)
                    except Exception as save_error:
                        print(f"Error saving after error in response {j}: {save_error}")
            
            # Final save for this qid (redundant but ensures consistency)
            result_to_write[qid] = qid_result
        else:
            print(f"No context found for qid {qid}")
            result_to_write[qid] = {
                "problem": query,
                "context": "No context found",
                "responses": {"error": "No context found"}
            }
            # Save no-context result
            try:
                experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)
            except Exception as e:
                print(f"Error saving no-context result: {e}")
        
    del llm
