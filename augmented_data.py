import json
import random
import re
from typing import List, Set, Tuple

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

nlp = spacy.load("en_core_web_sm")
stopwords = set(stopwords.words('english'))

# Initialize the Marian tokenizer and model for back-translation (English to French and back to English)
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
reverse_tokenizer = MarianTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-fr-en")
reverse_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

# Initialize the MLM pipeline (Masked Language Model)
mlm_pipeline = pipeline(
    'fill-mask', model='distilroberta-base', tokenizer='distilroberta-base')

# Load models and tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

sia = SentimentIntensityAnalyzer()


def sentiment_inversion(text: str) -> str:
    """
    Invert the sentiment of the input text using a T5 transformer model.

    Parameters:
    text (str): The input text to be processed.

    Returns:
    str: The text with inverted sentiment.
    """
    sentiment_score = sia.polarity_scores(text)['compound']
    if sentiment_score >= 0:
        task = "translate English to English and make it negative: "
    else:
        task = "translate English to English and make it positive: "

    t5_input = task + text
    input_ids = t5_tokenizer.encode(t5_input, return_tensors="pt")
    outputs = t5_model.generate(input_ids)
    inverted_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return inverted_text


def paraphrasing(text: str) -> str:
    """
    Paraphrase the input text using a T5 transformer model.

    Parameters:
    text (str): The input text to be paraphrased.

    Returns:
    str: The paraphrased text.
    """
    t5_input = "paraphrase: " + text
    input_ids = t5_tokenizer.encode(t5_input, return_tensors="pt")
    outputs = t5_model.generate(input_ids)
    paraphrased_text = t5_tokenizer.decode(
        outputs[0], skip_special_tokens=True)

    return paraphrased_text


def random_text_generation(prompt: str, length: int = 50) -> str:
    """
    Generate random text based on the given prompt using GPT-2.

    Parameters:
    prompt (str): The input prompt for text generation.
    length (int): The length of the generated text. Default is 50 tokens.

    Returns:
    str: The generated text.
    """
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(input_ids, max_length=length)
    generated_text = gpt2_tokenizer.decode(
        outputs[0], skip_special_tokens=True)

    return generated_text

def back_translation(text: str) -> str:
    encoded_text = tokenizer.encode(text, return_tensors="pt")
    translation = model.generate(encoded_text)
    decoded_translation = tokenizer.decode(
        translation[0], skip_special_tokens=True)
    reverse_encoded_translation = reverse_tokenizer.encode(
        decoded_translation, return_tensors="pt")
    reverse_translation = reverse_model.generate(reverse_encoded_translation)
    decoded_reverse_translation = reverse_tokenizer.decode(
        reverse_translation[0], skip_special_tokens=True)
    return decoded_reverse_translation


# Implement Synonym Replacement from EDA
def synonym_replacement(sentence: str, n: int = 1) -> str:
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()
    random_word_list = list(
        set([word for word in words if word not in stopwords]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word ==
                         random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)


# Implement Random Insertion from EDA
def random_insertion(sentence: str, n: int = 1) -> str:
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    return ' '.join(new_words)


def add_word(new_words: List[str]):
    synonyms = []
    counter = 0

    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return

    random_synonym = random.choice(list(synonyms))
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


# Implement Random Deletion from EDA
def random_deletion(sentence: str, p: float = 0.1) -> str:
    words = nltk.word_tokenize(sentence)

    if len(words) == 1:
        return words[0]

    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        random_idx = random.randint(0, len(words)-1)
        return words[random_idx]

    return ' '.join(new_words)

# Implement Random Swap from EDA


def random_swap(sentence: str, n: int = 1) -> str:
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()

    for _ in range(n):
        new_words = swap_word(new_words)

    return ' '.join(new_words)


def swap_word(words: List[str]):
    random_idx_1 = random.randint(0, len(words)-1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(words)-1)
        counter += 1
        if counter > 3:
            return words

    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    return words


# Implement MLM-based augmentation
def mlm_augmentation(sentence: str, n: int = 1) -> str:
    words = nltk.word_tokenize(sentence)
    augmented_sentence = sentence

    for _ in range(n):
        masked_index = random.randint(0, len(words) - 1)
        masked_word = words[masked_index]
        masked_sentence = ' '.join(
            [words[i] if i != masked_index else mlm_pipeline.tokenizer.mask_token for i in range(
                len(words))]
        )

        predictions = mlm_pipeline(masked_sentence)
        for prediction in predictions:
            if prediction['token_str'].strip() != masked_word:
                augmented_sentence = augmented_sentence.replace(
                    masked_word, prediction['token_str'].strip(), 1
                )
                break

    return augmented_sentence


def attribute_swapping(text: str, attributes: List[Tuple[str, str]]) -> str:
    for old_attribute, new_attribute in attributes:
        text = text.replace(old_attribute, new_attribute)
    return text


def random_attribute_generation(text: str, attributes: List[str], n: int = 1) -> str:
    for _ in range(n):
        random_attribute = random.choice(attributes)
        insert_position = random.randint(0, len(text))
        text = text[:insert_position] + \
            random_attribute + text[insert_position:]
    return text


def get_synonyms(word: str) -> Set[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').replace('-', ' ').lower()
            if synonym != word:
                synonyms.add(synonym)
    return synonyms


def validate_prompt_completion(prompt: str, completion: str) -> bool:
    if len(prompt) < 5 or len(completion) < 5:
        return False

    if not re.match(r'^[A-Z]', prompt) or not re.match(r'^[A-Z]', completion):
        return False

    return True


def entity_swapping(text: str) -> str:
    doc = nlp(text)
    entities = [ent for ent in doc.ents]
    if len(entities) < 2:
        return text

    i, j = random.sample(range(len(entities)), 2)
    text = text.replace(entities[i].text, '[[' + entities[j].text + ']]', 1)
    text = text.replace(entities[j].text, '[[' + entities[i].text + ']]', 1)
    text = text.replace('[[' + entities[j].text + ']]', entities[i].text)
    text = text.replace('[[' + entities[i].text + ']]', entities[j].text)
    return text


def entity_insertion_deletion(text: str, random_entities: List[str]) -> str:
    if not random_entities:
        return text

    doc = nlp(text)
    entities = [ent for ent in doc.ents]
    if not entities:
        return text

    random_entity = random.choice(random_entities)
    insert_position = random.randint(0, len(text))
    text = text[:insert_position] + random_entity + text[insert_position:]

    entity_to_delete = random.choice(entities)
    text = text.replace(entity_to_delete.text, "", 1)
    return text


def perform_data_augmentation(input_data: str, output_file_path: str, augmentation_functions: List[str]):
    with open(input_data, 'r') as file:
        data = [json.loads(line) for line in file]

    with open(output_file_path, 'w') as output_file:
        for item in data:
            print(f"Item: {item}")
            prompt, completion = item['prompt'], item['completion']

            if not validate_prompt_completion(prompt, completion):
                print(
                    f"Skipping invalid prompt-completion pair: {prompt} | {completion}")
                continue

            output_file.write(json.dumps(
                {"prompt": prompt, "completion": completion}) + '\n')

            for func_name in augmentation_functions:
                func = globals()[func_name]

                if func_name == 'text_mix':
                    continue  # Skip text_mix for now

                augmented_prompt = func(prompt)
                augmented_completion = func(completion)
                if validate_prompt_completion(augmented_prompt, augmented_completion):
                    output_file.write(json.dumps(
                        {"prompt": augmented_prompt, "completion": augmented_completion}) + '\n')
                else:
                    print(
                        f"Skipping invalid augmented prompt-completion pair ({func_name}): {augmented_prompt} | {augmented_completion}")

        print(f"Data augmentation and validation completed.")

# Main
if __name__ == "__main__":
    input_file = "jsonl/data.jsonl"
    output_file = "download/augmented_data.jsonl"
    augmentation_functions = [
        'back_translation',
        'synonym_replacement',
        'random_insertion',
        'random_deletion',
        'random_swap',
        'mlm_augmentation',
        'attribute_swapping',
        'random_attribute_generation',
        'sentiment_inversion',
        'paraphrasing',
        'random_text_generation',
        'entity_swapping',
        'entity_insertion_deletion'
    ]

    perform_data_augmentation(input_file, output_file, augmentation_functions)
