import os, csv, json
import argparse
import time
import sys
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
import tiktoken
import torch.multiprocessing as mp
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

load_dotenv()
DATADIR="./data"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

TongyiAPI_thinking_models=["qwen3-max-preview"]
OpenRouter_thinking_models = ["anthropic/claude-sonnet-4.5", "google/gemini-3-flash-preview", "openai/gpt-5.2", "openai/gpt-oss-120b"]
Non_thinking_models = ["openai/gpt-4o-mini", "meta-llama/llama-4-maverick", "qwen/qwen3-235b-a22b-2507"]

template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()

THINKING_EFFORT='medium'
THINKING_BUDGET = 2048
GEN_BUDGET = 256

class AnswerResponseBinary_cot(BaseModel):
    analysis: str 
    answer: Literal['A', 'B']

class AnswerResponseMultiple_cot(BaseModel):
    analysis: str 
    answer: Literal['A', 'B', 'C', 'D']

class AnswerResponseBinary_0shot(BaseModel):
    answer: Literal['A', 'B']

class AnswerResponseMultiple_0shot(BaseModel):
    answer: Literal['A', 'B', 'C', 'D']

def get_response_format(task: str, cot: bool):
    if cot:
        schema = AnswerResponseBinary_cot.model_json_schema() if task in ['task1', 'task2', 'task4'] else AnswerResponseMultiple_cot.model_json_schema()
    else:
        schema = AnswerResponseBinary_0shot.model_json_schema() if task in ['task1', 'task2', 'task4'] else AnswerResponseMultiple_0shot.model_json_schema()
    
    # Azure OpenAI requires additionalProperties to be explicitly set to false
    schema["additionalProperties"] = False
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_response", 
            "strict": True,  
            "schema": schema
        }
    }

def setup_file_logging(model: str, task: str):
    """Setup file logging with model and task in filename."""
    model_name = model.split("/")[-1]  
    log_filename = os.path.join(LOG_DIR, f"{model_name}_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.add(
        log_filename,
        rotation="100 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,  
    )
    logger.info(f"Logging to file: {log_filename}")


def get_api_config(model: str) -> tuple[str, str]:
    model_lower = model.lower()
    model_prefixes = ['gpt', 'llama', 'qwen', 'deepseek', 'claude', 'gemini']
    prefix = 'gpt'  
    for keyword in model_prefixes:
        if keyword in model_lower:
            prefix = keyword
            break
    if model == "qwen3-235b":
        prefix = 'gpt'  

    base_url = os.getenv(f"{prefix}_base_url")
    api_key = os.getenv(f"{prefix}_api_key")
    
    if not base_url or not api_key:
        logger.warning(f"❗️❗️❗️ API config for '{prefix}' not found, falling back to 'openai'")
        base_url = os.getenv("gpt_base_url")
        api_key = os.getenv("gpt_api_key")
    
    logger.info(f"Using API config: prefix='{prefix}', base_url='{base_url}'")
    return base_url, api_key


def query_llm(prompt, model, tokenizer, item, client, args, temperature=0.1):
    max_len = maxlen_map[model]

    input_ids = tokenizer.encode(prompt, disallowed_special=())
    assert len(input_ids) <= max_len, f"Input length {len(input_ids)} exceeds max length {max_len}"

    tries = 0
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        try:
            if model in TongyiAPI_thinking_models:
                if args.cot:
                    extra_body = {
                        "enable_thinking": True,
                        "thinking_budget": THINKING_BUDGET,
                    }
                    max_tokens = GEN_BUDGET
                else:
                    extra_body = {
                        "enable_thinking": False,
                    }
                    max_tokens = GEN_BUDGET          

                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    extra_body=extra_body,
                    stream=True,
                    stream_options={"include_usage": True},
                    max_tokens=max_tokens
                )
                
                reasoning_text = ""
                answer_content = ""
                latest_status = "think"
                tokens = {}
                
                for chunk in completion:
                    if chunk.choices == []:
                        print(chunk)
                        tokens["total_tokens"] = chunk.usage.total_tokens
                        tokens["prompt_tokens"] = chunk.usage.prompt_tokens
                        tokens["completion_tokens"] = chunk.usage.completion_tokens
                        tokens["reasoning_tokens"] = chunk.usage.completion_tokens_details.reasoning_tokens if hasattr(chunk.usage.completion_tokens_details, 'reasoning_tokens') else 0
                        continue
                    if chunk.choices[0].delta.content is not None:
                        if latest_status == "think":
                            latest_status = "answer"
                        answer_content += chunk.choices[0].delta.content
                    else:
                        reasoning_text += chunk.choices[0].delta.reasoning_content
                
                response_text = f"<Internal Reasoning>\n{reasoning_text}\n</Internal Reasoning>\n\n"
                
                try:
                    json_content = answer_content.strip()
                    start_idx = json_content.find('{')
                    end_idx = json_content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_content = json_content[start_idx:end_idx + 1]
                    
                    parsed = json.loads(json_content)
                    analysis = parsed.get('analysis', '')
                    pred = parsed.get('answer', '')
                    response_text += f"<Analysis>\n{analysis}\n</Analysis>\n\n<Answer>\nThe correct answer is ({pred}).\n</Answer>"
                except Exception as e:
                    logger.warning(f"Failed to parse JSON response for {model}, using raw content: {e}")
                    response_text += answer_content

                logger.info(f"Response for {model}: {response_text}")

            elif model in OpenRouter_thinking_models:
                if args.cot:
                    max_tokens = THINKING_BUDGET+GEN_BUDGET
                else:
                    max_tokens = GEN_BUDGET
                if model in ["anthropic/claude-sonnet-4.5"]:
                    extra_body = {
                        "reasoning": {
                            "enabled": True if args.cot else False,
                            "max_tokens": THINKING_BUDGET if args.cot else 0,
                        }
                    }
                
                elif model in ["openai/gpt-oss-120b", "google/gemini-3-flash-preview", "openai/gpt-5.2"]:
                    extra_body = {
                        "reasoning": {
                            "enabled": True if args.cot or 'gpt' in model else False,
                            "effort": THINKING_EFFORT if args.cot else 'none',
                        }
                    }
                    if model == "openai/gpt-oss-120b" and not args.cot:
                        extra_body["reasoning"]["effort"] = "low"
                
                else:
                    raise ValueError(f"Not implemented model: {model}")

                response_format_config = get_response_format(args.task, args.cot)
                
                extra_body["response_format"] = response_format_config
                if "provider" not in extra_body:
                    extra_body["provider"] = {}    
                extra_body["provider"]["ignore"] = ["siliconflow/fp8",'fireworks','cerebras']

                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    extra_body=extra_body,
                    stream=False,
                    max_tokens=max_tokens
                )
                
                response_text = ""
                tokens = {}
                
                if completion.usage:
                    tokens["total_tokens"] = completion.usage.total_tokens
                    tokens["prompt_tokens"] = completion.usage.prompt_tokens
                    tokens["completion_tokens"] = completion.usage.completion_tokens
                    tokens['provider'] = completion.provider
                    if hasattr(completion.usage, 'completion_tokens_details') and completion.usage.completion_tokens_details:
                        tokens["reasoning_tokens"] = getattr(completion.usage.completion_tokens_details, 'reasoning_tokens', 0)
                
                if hasattr(completion.choices[0].message, 'reasoning') and completion.choices[0].message.reasoning:
                    response_text = f"<Internal Reasoning>\n{completion.choices[0].message.reasoning}\n</Internal Reasoning>\n\n"
                elif model in ["openai/gpt-5.2"] and hasattr(completion.choices[0], 'reasoning_details') and len(completion.choices[0].reasoning_details) > 0 and hasattr(completion.choices[0].reasoning_details[0], 'summary') and completion.choices[0].reasoning_details[0].summary:
                    response_text = f"<Internal Reasoning>\n{completion.choices[0].reasoning_details[0].summary}\n</Internal Reasoning>\n\n"
                
                if completion.choices[0].message.content:
                    try:
                        analysis = json.loads(completion.choices[0].message.content).get('analysis')
                        pred = json.loads(completion.choices[0].message.content).get('answer')
                        response_text += f"<Analysis>\n{analysis}\n</Analysis>\n\n<Answer>\nThe correct answer is ({pred}).\n</Answer>"
                    except Exception as e:
                        logger.error(f"Failed to parse structured output response for {model}: {e}")
                        response_text += completion.choices[0].message.content

                logger.info(f"Structured output response for {model}: {response_text}")

            elif model in Non_thinking_models:
                if args.cot:
                    max_tokens = THINKING_BUDGET+GEN_BUDGET
                else:
                    max_tokens = GEN_BUDGET

                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=False,
                    max_tokens=max_tokens,
                    response_format=get_response_format(args.task, args.cot)
                )
                
                tokens = {
                    "total_tokens": completion.usage.total_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                }
                
                if hasattr(completion.usage, 'completion_tokens_details') and completion.usage.completion_tokens_details:
                    tokens["reasoning_tokens"] = getattr(completion.usage.completion_tokens_details, 'reasoning_tokens', 0)
                
                response_text = ""
                if completion.choices[0].message.content:
                    try:
                        parsed = json.loads(completion.choices[0].message.content)
                        analysis = parsed.get('analysis', '')
                        pred = parsed.get('answer', '')
                        response_text = f"<Analysis>\n{analysis}\n</Analysis>\n\n<Answer>\nThe correct answer is ({pred}).\n</Answer>"
                    except Exception as e:
                        logger.error(f"Failed to parse structured output response: {e}")
                        response_text = completion.choices[0].message.content
                
                logger.info(f"Structured output response: {response_text}")

            else: 
                raise ValueError(f"Unimplemented model: {model}")
                
            return response_text, tokens
                
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if hasattr(e, 'response'):
                logger.error(f"Error Occurs: {item['id']} {str(e)}. Response content: {e.response.text}")
            else:
                logger.error(f"Error Occurs: {item['id']} {str(e)}")
            time.sleep(3)
    else:
        logger.error("Max tries. Failed.")
        return '', {}


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    
    match = re.search(r'The correct answer is ([A-D])', response)
    if match:
        return match.group(1)
    
    try:
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([A-D])"[^{}]*\}', response)
        if json_match:
            return json_match.group(1)
        
        start_idx = response.rfind('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx + 1]
            parsed = json.loads(json_str)
            if 'answer' in parsed:
                return parsed['answer']
    except Exception as e:
        logger.warning(f"Failed to extract answer from JSON in response: {e}")
    
    return None


def get_pred(data, args, out_file, rank):
    model = args.model
    tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    base_url, api_key = get_api_config(model)
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    temp_file = f"{out_file}.tmp.{rank}"
    with open(temp_file, 'w', encoding='utf-8') as fout:
        for item in tqdm(data, desc=f"Process {rank}"):
            sequence = item['sequence']

            if args.cot:
                template = template_0shot_cot
            else:
                template = template_0shot

            if args.task in ['task1', 'task2', 'task4']:
                options = "A, B"
            else:
                options = "A, B, C, D"

            prompt = template.replace('$sequence$', sequence.strip()).replace('$question$', item['question'].strip()).replace('$choices$', item['choices']).replace('$options$', options)
            
            # 调用 LLM
            output, tokens = query_llm(prompt, model, tokenizer, item, client, args, temperature=0.1)
            if output == '':
                continue

            response = output.strip()
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            item['sequence'] = sequence[:100]
            item['tokens'] = tokens
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()


def merge_results(out_file, n_proc):
    with open(out_file, 'a', encoding='utf-8') as fout:
        for rank in range(n_proc):
            temp_file = f"{out_file}.tmp.{rank}"
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(temp_file)  
    logger.info(f"Merged results from {n_proc} processes to {out_file}")


def main():
    logger.info(args)

    if args.cot:
        out_file = os.path.join(args.save_dir, args.task, "0shot_cot", args.model.split("/")[-1] + f"_{args.task}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.task, "0shot", args.model.split("/")[-1] + f"_{args.task}.jsonl")
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    data_all = json.load(open(f'{DATADIR}/{args.task}.json', 'r', encoding='utf-8'))

    if args.num_samples > len(data_all):
        logger.warning("❗️❗️❗️ num_samples is larger than the number of samples in the dataset. Loading all samples.")
    elif args.num_samples > 0:
        logger.info(f"Loading {args.num_samples} samples.")
        data_all = data_all[:args.num_samples]
    elif args.num_samples <= 0:
        logger.info("Loading all samples.")

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["id"]: 0 for line in f}
    
    data = []
    for item in data_all:
        if item["id"] not in has_data:
            data.append(item)
    
    if not data:
        logger.success("🎉🎉🎉 All samples have been processed. Nothing to do.")
        return
    
    logger.info(f"🏃🏃‍🏃 Processing {len(data)} samples with {args.n_proc} processes...")
    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        if data_subsets[rank]: 
            p = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file, rank))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
    
    merge_results(out_file, args.n_proc)
    logger.success(f"🎉🎉🎉 Finished the eval of {args.task} with {args.model}, saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="gpt-oss", choices=model_map.keys())
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--num_samples", "-ns", type=int, default=1)
    parser.add_argument("--task", "-t", type=str, required=True, choices=["task1", "task2", "task3", "task4", "task5"])
    args = parser.parse_args()

    assert args.task in ["task1", "task2", "task3", "task4", "task5"], f"Invalid task {args.task}. Valid tasks: [task1, task2, task3, task4, task5]"
    assert args.model in model_map, f"Invalid model {args.model}. Valid models: {model_map.keys()}"
    assert args.n_proc > 0, f"Invalid number of processes {args.n_proc}."

    setup_file_logging(args.model, args.task)
    main()