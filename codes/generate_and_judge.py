import json
import argparse
import os
from pathlib import Path
from openai import OpenAI
from vllm import LLM, SamplingParams
import re
import requests

StatusCodeWj  = "WJ"
StatusCodeAc  = "AC"
StatusCodeWa  = "WA"
StatusCodeRe  = "RE"
StatusCodeTle = "TLE"
StatusCodeMle = "MLE"
StatusCodeDle = "DLE" # Depth Limit Exceeded
StatusCodeUge = "UGE" # Unauthorized Gate Error
StatusCodeQle = "QLE" # Qubits Limit Exceeded
StatusCodeUme = "UME"

base_dir = Path(__file__).parent
current_dir = Path(__file__).parent

def call_openai_api(model_name: str, messages: list, client) -> str:
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
        )
        response_text = chat_response.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        print(f"API call failed: {e}")
        return ""


def generate_with_local_model(llm, messages: list) -> str:
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        stop=None
    )
    outputs = llm.chat(messages, sampling_params=sampling_params)
    response_text = outputs[0].outputs[0].text
    return response_text


def parse_args():
    '''
    You can use a variety of models other than those listed here.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    #parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    #parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    #parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    #parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    return parser.parse_args()

def check_code_exists(response_text, file_name, messages, payload, args, client, is_openai_api):
    code_exists = False
    for i in range(3):
        code_exists = extract_last_python_block_to_folder(response_text, file_name.replace(".txt", ".py"))
        if not code_exists:
            print("Code does not exist.")
            payload["messages"].append({
                "role": "system",
                "content": ""
            })
            payload["messages"].append({
                "role": "user",
                "content": f"Your answer doesn't include <'''python>~<'''>. Try again."
            })
            if is_openai_api:
                response_text = call_openai_api(args.model, messages, client)
            else:
                response_data = requests.post("http://localhost:8800/v1", json=payload).text
                data = json.loads(response_data)
                print(data)
                response_text = data["choices"][0]["message"]["content"]
    return code_exists

def extract_last_python_block_to_folder(text: str, file_name: str):
    '''
    Check whether markdown expression is included or not.
    '''

    pattern = r"```python\s+(.*?)```|```python\s+(.*?)'''"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return False

    # If the model generated multiple answer, the last one will be taken.
    last_match = matches[-1]
    code = last_match[0] if last_match[0] else last_match[1]

    # Response is recorded to the folder 'Responses'.
    target_dir = Path(__file__).parent.joinpath("Responses")
    output_path = target_dir / file_name
    output_path.write_text(code.strip(), encoding='utf-8')
    return True

def main():
    args = parse_args()

    with open(current_dir.joinpath("keys.json"), "r") as f:
        keys = json.load(f)
    api_key = keys["api_key"]
    api_base = keys.get("api_base", "http://localhost:8800/v1")

    client = None
    llm = None
    if "gpt-" in args.model or "o3" in args.model in args.model:
        print('Using OpenAI')
        print(args.model)
        client = OpenAI(api_key=api_key)
        base_url = api_base
        is_openai_api = True
    else:
        llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=8)
        # Do not forget to change max_tokens in generate_with_local_model()
        is_openai_api = False

    folder_path = base_dir.joinpath("Problems")
    problem_list = sorted(os.listdir(folder_path))

    file_path = base_dir.joinpath("rules.txt")
    with open(file_path,"r",encoding="utf-8") as f:
        rules = f.read()
    output_data = {}
    for file_name in problem_list:
        # The problems assigned to the model can be controlled through slicing. e.g. for file_name in problem_list[4:10]:
        file_path = os.path.join(folder_path, file_name)

        with open(file_path,"r") as f:
            problem_statement = ''.join(f.readlines())

        qcoder_question = rules + problem_statement

        messages = [
            {
                "role": "system",
                "content": "You are an AI coding agent for a quantum programming language."
            },
            {
                "role": "user",
                "content": f"{qcoder_question}"
            }
        ]

        payload = {}
        if not is_openai_api:
            payload = {
                "model": f"{args.model}",
                "messages": messages
            }

        ans_iter = 1
        statuscode_hist = []
        max_refinement_iter = 5
        for iter in range(max_refinement_iter):

            if is_openai_api:
                response_text = call_openai_api(args.model, messages, client)
            else:
                response_data = requests.post(api_base, json=payload).text
                data = json.loads(response_data)
                print(data)
                response_text = data["choices"][0]["message"]["content"]

            print("-----------------------------------------------------")
            print(response_text)
            print("-----------------------------------------------------")

            code_exists = check_code_exists(response_text, file_name, messages,payload, args, client, is_openai_api)

            if not code_exists:
                statuscode = 'CNE'
            else:
                contestID, problemID = file_name.replace('.txt','').split("_")
                url = f"https://qcoder.jp/api/research/contests/{contestID}/problems/{problemID}"

                with open(base_dir.joinpath("QCoder_cookie.txt"),"r",encoding="utf-8") as f:
                    session_cookie = f.read()

                headers = {
                    "Content-Type": "application/json",
                    "Cookie": f"session={session_cookie}"
                }

                with (open(base_dir.joinpath("Responses").joinpath(file_name.replace('txt','py')),'r',encoding='utf-8') as f):
                    code = ''.join(f.readlines())

                data = {
                    "sourceCode": f"{code}"
                }

                response = requests.post(url, headers=headers, json=data)

                result = json.loads(response.text)
                test_results = result["testResults"]

                for test in test_results:
                    memory = test.get("memory", None)
                    statuscode = test.get("statusCode", "")
                    time = test.get("time", None)
                    error_text = test.get("errorMessage", "")


                print(statuscode)
                statuscode_hist.append(statuscode)

                if statuscode == StatusCodeAc:
                    break
                else:
                    answer = ""
                    with open(os.path.join(base_dir.joinpath("Responses"), file_name.replace('txt','py'))) as f:
                        answer += (''.join(f.readlines()))
                    messages.append({
                        "role": "system",
                        "content": answer
                    })

                    if statuscode == StatusCodeWa:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{"Your answer was\n\n '''python\n" + answer + "'''\n\n and this is wrong. Try again."}"
                            }
                        )
                    elif statuscode == StatusCodeTle:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{"Your answer was\n\n '''python\n" + answer + "'''\n\n and the execution time exceeded. Please revise your implementation to improve efficiency. Try again."}"
                            }
                        )
                    elif statuscode == StatusCodeDle:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{"Your answer was\n\n '''python\n" + answer + "'''\n\n and The circuit depth exceeded the given constraint. Please revise your implementation to improve efficiency. Try again."}"
                            }
                        )
                    elif statuscode == StatusCodeUme:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{"Your answer was\n\n '''python\n" + answer + f"'''\n\n and unauthorized module has been used (error text: {error_text}. Try again."}"
                            }
                        )
                    elif statuscode == StatusCodeUge:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{"Your answer was\n\n '''python\n" + answer + f"'''\n\n and unauthorized quantum gate has been used (error text: {error_text}). Try again."}"
                            }
                        )
                    else:
                        error_text = '\n'.join(re.findall(r"[A-Za-z]*Error:.*", error_text))
                        print(error_text)
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{"Your answer was\n\n '''python\n" + answer + "'''\n\n and the occurring error is" + error_text + ". Try again."}"
                            }
                        )
            if iter != 5:
                ans_iter += 1

        print(file_name + ': ' + statuscode)
        output_data[file_name.replace('.txt', '')] = '[' + ','.join(statuscode_hist) + ': ' + str(ans_iter) + ']'
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Call LLM via API or vllm
    # response_text = call_openai_api(args.model, messages, client)


if __name__ == "__main__":
    main()
