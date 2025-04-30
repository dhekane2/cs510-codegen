import re
import json
import argparse
import os

def post_process_llama_output(raw_text):
    python_pattern = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    python_block_matches = python_pattern.findall(raw_text)
    if python_block_matches:
      target_code = "\n\n".join(code.strip() for code in python_block_matches)
      version = "python3"
      resp = "```json\n[" + json.dumps({"version": version,"target code": target_code}) + "]\n```"
      return resp
      
    list_start = raw_text.find('[{"version')
    if list_start:

      list_text = raw_text[list_start:]
      list_end = list_text.find('}]\n') + 1
      list_text = list_text[:list_end]

      result = f"```json\n{list_text}\n```"
      return result
    
    return f"```json\n\n```"


def main():
    parser = argparse.ArgumentParser(description="Post-process LLaMA program synthesis results.")
    parser.add_argument("--inference_dir", type=str, required=True)
    parser.add_argument("--postprocessed_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    inference_path = os.path.join(args.inference_dir, f"program_synthesis_eval_{args.model_name}.jsonl")
    postprocessed_write_path = os.path.join(args.postprocessed_dir, f"program_synthesis_eval_{args.model_name}.jsonl")

    inference_data = []
    with open(inference_path, "r", encoding="utf-8") as infile:
        for line in infile:
          inference_data.append(json.loads(line))

    with open(postprocessed_write_path, "w", encoding="utf-8") as outfile:
        for response in inference_data:
            try:
                processed_code_dict = {
                    "lang_cluster": response.get('lang_cluster', ''),
                    "lang": response.get('lang', ''),
                    "src_uid": response.get('src_uid', ''),
                    "difficulty": response.get('difficulty', ''),
                    "testcases": response.get('testcases', ''),
                }

                for i in range(6):
                    src_code_key = f'program_synthesis_{i}'
                    if src_code_key in response and response[src_code_key]:
                        processed_code_dict[src_code_key] = post_process_llama_output(response[src_code_key])

                outfile.write(json.dumps(processed_code_dict) + '\n')

            except Exception as e:
                print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
