import json
import argparse

def main(results_path, exec_results_path, model_name):
    HARD_BAR = 1501
    NON_BAR = 2701

    easy_total = 0
    hard_total = 0

    # Read results and count total Easy and Hard problems
    with open(results_path, "r", encoding="utf-8") as responses:
        for response in responses:
            problem = json.loads(response)
            difficulty = problem['difficulty']
            if difficulty < HARD_BAR:
                easy_total += 1
            elif HARD_BAR <= difficulty < NON_BAR:
                hard_total += 1

    easy_solved = 0
    hard_solved = 0

    # Read execution results and count solved Easy and Hard problems
    with open(exec_results_path, "r", encoding="utf-8") as responses:
        for response in responses:
            problem = json.loads(response)
            difficulty = problem['difficulty']
            if difficulty < HARD_BAR:
                easy_solved += 1
            elif HARD_BAR <= difficulty < NON_BAR:
                hard_solved += 1

    # Compute Pass@5 scores
    easy_pass_at_5 = round((easy_solved / easy_total) * 100, 2) if easy_total > 0 else 0.0
    hard_pass_at_5 = round((hard_solved / hard_total) * 100, 2) if hard_total > 0 else 0.0

    # Prepare and print the results
    results_summary = {
        "Model": [model_name],
        "Easy": [easy_pass_at_5],
        "Hard": [hard_pass_at_5],
    }

    print(results_summary)
    #df = pd.DataFrame(results_summary)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pass@5 for Easy and Hard problems.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results JSONL file.")
    parser.add_argument("--exec_results_path", type=str, required=True, help="Path to the execution results JSONL file.")
    parser.add_argument("--model_name", type=str, required=True, help="give model name here.")


    args = parser.parse_args()
    main(args.results_path, args.exec_results_path,args.model_name)
