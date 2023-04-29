import sys
import os
import spacy

nlp = spacy.load("en_core_web_sm")


def segment_and_extract(input_text: str):
    prompt_completion_pairs = extract_prompt_completion_pairs(input_text)
    extracted_text = ""

    for pair in prompt_completion_pairs:
        extracted_text += f"prompt: {pair['prompt']}\n"
        extracted_text += f"completion: {pair['completion']}\n"
        extracted_text += "\n"

    return extracted_text


def extract_prompt_completion_pairs(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    prompt_completion_pairs = []

    for i in range(len(sentences) - 1):
        prompt_completion_pairs.append({
            'prompt': sentences[i],
            'completion': sentences[i + 1]
        })

    return prompt_completion_pairs
