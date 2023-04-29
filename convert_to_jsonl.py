# convert_to_jsonl.py
import json
import openai
import os
from typing import List, Callable


def save_jsonl_classification(prompt_completion_pairs: List[dict], output_file_path: str) -> None:
    with open(output_file_path, 'w') as f:
        for pair in prompt_completion_pairs:
            f.write(json.dumps(pair) + '\n')


def save_jsonl_sentiment_analysis(prompt_completion_pairs: List[dict], output_file_path: str) -> None:
    with open(output_file_path, 'w') as f:
        for pair in prompt_completion_pairs:
            f.write(json.dumps(pair) + '\n')


def save_jsonl_entity_extraction(prompt_completion_pairs: List[dict], output_file_path: str) -> None:
    with open(output_file_path, 'w') as f:
        for pair in prompt_completion_pairs:
            f.write(json.dumps(pair) + '\n')


def save_jsonl_chatbot(prompt_completion_pairs: List[dict], output_file_path: str) -> None:
    with open(output_file_path, 'w') as f:
        for pair in prompt_completion_pairs:
            f.write(json.dumps(pair) + '\n')


def save_jsonl_product_description(prompt_completion_pairs: List[dict], output_file_path: str) -> None:
    with open(output_file_path, 'w') as f:
        for pair in prompt_completion_pairs:
            f.write(json.dumps(pair) + '\n')


def parse_prompt_completion_pairs(input_file_path: str) -> List[dict]:
    with open(input_file_path, 'r') as f:
        input_text = f.read()

    lines = input_text.strip().split('\n')
    prompt_completion_pairs = []

    for line in lines:
        prompt, completion = line.split('|', 1)
        prompt_completion_pairs.append({
            'prompt': prompt.strip(),
            'completion': completion.strip()
        })

    return prompt_completion_pairs


if __name__ == '__main__':
    input_file_path = os.path.join('input', 'input.txt')
    output_file_path = os.path.join('output', 'output.jsonl')

    prompt_completion_pairs = parse_prompt_completion_pairs(input_file_path)

    # Choose the appropriate save_jsonl_* function depending on the use case
    save_jsonl_classification(prompt_completion_pairs, output_file_path)
