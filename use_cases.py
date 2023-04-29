import os
import json
from typing import List, Callable
from text_preprocessor import preprocess_text
from segment_extract_text import segment_and_extract
from convert_to_jsonl import (
    save_jsonl_classification as convert_to_jsonl_classification,
    save_jsonl_sentiment_analysis as convert_to_jsonl_sentiment_analysis,
    save_jsonl_entity_extraction as convert_to_jsonl_entity_extraction,
    save_jsonl_chatbot as convert_to_jsonl_chatbot,
    save_jsonl_product_description as convert_to_jsonl_product_description,
)

from augmented_data import (perform_data_augmentation,
                            synonym_replacement,
                            random_insertion,
                            back_translation,
                            sentiment_inversion,
                            paraphrasing,
                            random_text_generation,
                            entity_swapping,
                            entity_insertion_deletion,
                            random_deletion,
                            random_swap,
                            mlm_augmentation,
                            attribute_swapping,
                            random_attribute_generation)


def read_text_from_file(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        content = f.read()
    decoded_content = content.decode('utf-8', errors='ignore')
    cleaned_content = decoded_content.replace('\x00', '')
    return cleaned_content


def create_output_file_path(app_root_path: str, prefix: str) -> str:
    output_dir = os.path.join(app_root_path, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, f"{prefix}_output.jsonl")


def save_text_to_file(text: str, file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def text_classification_pipeline(app_root_path: str, input_file_path: str) -> str:
    text = read_text_from_file(input_file_path)
    preprocessed_text = preprocess_text(text, 'classification')
    prompt_completion_pairs = segment_and_extract(preprocessed_text)
    txt_file_path = create_output_file_path(
        app_root_path, "classification_txt")
    save_text_to_file(prompt_completion_pairs, txt_file_path)
    output_file_path = create_output_file_path(app_root_path, "classification")
    convert_to_jsonl_classification(txt_file_path, output_file_path)

    augmentation_functions = [synonym_replacement,
                              random_insertion, back_translation]
    perform_data_augmentation(
        output_file_path, output_file_path, augmentation_functions)
    return output_file_path


def sentiment_analysis_pipeline(app_root_path: str, input_file_path: str) -> str:
    text = read_text_from_file(input_file_path)
    preprocessed_text = preprocess_text(text, 'sentiment_analysis')
    prompt_completion_pairs = segment_and_extract(preprocessed_text)
    txt_file_path = create_output_file_path(
        app_root_path, "sentiment_analysis_txt")
    save_text_to_file(prompt_completion_pairs, txt_file_path)
    output_file_path = create_output_file_path(
        app_root_path, "sentiment_analysis")
    convert_to_jsonl_sentiment_analysis(txt_file_path, output_file_path)

    augmentation_functions = [sentiment_inversion,
                              paraphrasing, random_text_generation]
    perform_data_augmentation(
        output_file_path, output_file_path, augmentation_functions)
    return output_file_path


def entity_extraction_pipeline(app_root_path: str, input_file_path: str) -> str:
    text = read_text_from_file(input_file_path)
    preprocessed_text = preprocess_text(text, 'entity_extraction')
    prompt_completion_pairs = segment_and_extract(preprocessed_text)
    txt_file_path = create_output_file_path(
        app_root_path, "entity_extraction_txt")
    save_text_to_file(prompt_completion_pairs, txt_file_path)
    output_file_path = create_output_file_path(
        app_root_path, "entity_extraction")
    convert_to_jsonl_entity_extraction(txt_file_path, output_file_path)

    augmentation_functions = [synonym_replacement,
                              entity_swapping, entity_insertion_deletion]
    perform_data_augmentation(
        output_file_path, output_file_path, augmentation_functions)
    return output_file_path


def chatbot_pipeline(app_root_path: str, input_file_path: str) -> str:
    text = read_text_from_file(input_file_path)
    preprocessed_text = preprocess_text(text, 'chatbot')
    prompt_completion_pairs = segment_and_extract(preprocessed_text)
    txt_file_path = create_output_file_path(app_root_path, "chatbot_txt")
    save_text_to_file(prompt_completion_pairs, txt_file_path)
    output_file_path = create_output_file_path(app_root_path, "chatbot")
    convert_to_jsonl_chatbot(txt_file_path, output_file_path)

    augmentation_functions = [
        synonym_replacement, random_insertion, random_deletion, random_swap, mlm_augmentation]
    perform_data_augmentation(
        output_file_path, output_file_path, augmentation_functions)
    return output_file_path


def product_description_pipeline(app_root_path: str, input_file_path: str) -> str:
    text = read_text_from_file(input_file_path)
    preprocessed_text = preprocess_text(text, 'product_description')
    prompt_completion_pairs = segment_and_extract(preprocessed_text)
    txt_file_path = create_output_file_path(
        app_root_path, "product_description_txt")
    save_text_to_file(prompt_completion_pairs, txt_file_path)
    output_file_path = create_output_file_path(
        app_root_path, "product_description")
    convert_to_jsonl_product_description(txt_file_path, output_file_path)

    augmentation_functions = [
        attribute_swapping, random_attribute_generation]
    perform_data_augmentation(
        output_file_path, output_file_path, augmentation_functions)
    return output_file_path
