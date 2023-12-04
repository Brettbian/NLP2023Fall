from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from functools import partial
import os
import ast

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        pass

def _get_max_length(model):
    """
    Get the maximum length setting from the model's configuration.

    Args:
        model: The model object.

    Returns:
        max_length (int): The maximum length setting.
    """
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def load_model(model_name, bnb_config):
    """
    Load the model and tokenizer.

    Args:
        model_name (str): The name of the model.
        bnb_config: The quantization configuration.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available resources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _identify_dataset_type(dataset_name):
    """
    Identify the type of dataset based on its name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        dataset_type (str): The type of dataset.
    """
    if bool(re.search(r'McTest', dataset_name, re.IGNORECASE)):
        return 'mctest'
    elif bool(re.search(r'RACE', dataset_name, re.IGNORECASE)):
        return 'race'
    elif bool(re.search(r'CommonsenseQA', dataset_name, re.IGNORECASE)):
        return 'commonsenseqa'
    elif bool(re.search(r'ARC', dataset_name, re.IGNORECASE)):
        return 'arc'

def get_answer_column(dataset_name):
    dataset_type = _identify_dataset_type(dataset_name)
    if dataset_type == 'mctest':
        return 'Actual Answer'
    if dataset_type == 'arc':
        return 'answerKey'
    if dataset_type == 'commonsenseqa':
        return 'answerKey'
    if dataset_type == 'race':
        return 'answer'

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding = True,
    )

def format_data(sample, dataset_name, cot):
    dataset_type = _identify_dataset_type(dataset_name)
    INSTRUCTION_KEY = "<s>[INST]"
    INSTRUCTION_END = "</INST>"
    END_OF_SENTENCE = "</s>"
    if dataset_type == 'mctest':
        #replace //newline with \n
        sample['Story'] = sample['Story'].replace('\\newline', '')
        prompt = f"Question: {sample['Question']} Based on the following article:\n{sample['Story']}\nOptions: \n{sample['Options']}"
        predicted_answer = sample['Predicted Answer']
        actual_answer = sample['Actual Answer']
        explaination = sample['Explanation']
    elif dataset_type == 'race':
        prompt = f"Question: {sample['question']} Based on the following article:\n{sample['article']}\nOptions: \n{sample['options']}"
        predicted_answer = sample['predicted_answer']
        actual_answer = sample['answer']
        explaination = sample['explanation']
    elif dataset_type =='commonsenseqa':
        choices = sample['choices']
        try:
            data_dict = ast.literal_eval(choices)
            combined_list = [f"{label}. {text}" for label, text in zip(data_dict['label'], data_dict['text'])]
            formatted_choices = " ".join(combined_list)
        except (SyntaxError, ValueError):
            formatted_choices = choices
        prompt = f"Question: {sample['question']} \nOptions: \n{formatted_choices}"
        predicted_answer = sample['predicted_answer']
        actual_answer = sample['answerKey']
        explaination = sample['explanation']
    elif dataset_type == 'arc':
        choices = sample['choices']
        try:
            data_dict = ast.literal_eval(choices)
            combined_list = [f"{label}. {text}" for label, text in zip(data_dict['label'], data_dict['text'])]
            formatted_choices = " ".join(combined_list)
        except (SyntaxError, ValueError):
            formatted_choices = choices
        prompt = f"Question: {sample['question']} \nOptions: \n{formatted_choices}"
        predicted_answer = sample['predicted_answer']
        actual_answer = sample['answerKey']
        explaination = sample['explanation']
    else:
        raise Exception("Dataset type not recognized.")
    if cot:
        if predicted_answer == actual_answer:
            sample['text'] = f"{INSTRUCTION_KEY}{prompt}{INSTRUCTION_END} Explanation: {explaination}{END_OF_SENTENCE}"
        else:
            sample['text'] = f"{INSTRUCTION_KEY}{prompt}{INSTRUCTION_END}{END_OF_SENTENCE}"
    else: 
        sample['text'] = f"{INSTRUCTION_KEY}{prompt}{INSTRUCTION_END} Answer: {actual_answer}{END_OF_SENTENCE}"
    return sample

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset, cot, dataset_name):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    _format_data = partial(format_data, dataset_name = dataset_name, cot = cot)
    dataset = dataset.map(_format_data)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Set the tensor type
    dataset.set_format("torch")

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    # Print one sample from dataset
    print("Sample from dataset:")
    print(dataset["train"][1])

    return dataset

def generate_prompt(sample, dataset_name):
    dataset_type = _identify_dataset_type(dataset_name)
    if dataset_type == 'mctest':
        sample['Story'] = sample['Story'].replace('\\newline', '')
        input = f"{sample['Question']} Based on the following article:\n{sample['Story']}\nOptions: \n{sample['Options']}"
    elif dataset_type == 'arc':
        choices = sample['choices']
        try:
            data_dict = ast.literal_eval(choices)
            combined_list = [f"{label}. {text}" for label, text in zip(data_dict['label'], data_dict['text'])]
            formatted_choices = " ".join(combined_list)
        except (SyntaxError, ValueError):
            formatted_choices = choices
        input = f"{sample['question']} \nOptions: \n{formatted_choices}"
    elif dataset_type == 'commonsenseqa':
        choices = sample['choices']
        try:
            data_dict = ast.literal_eval(choices)
            combined_list = [f"{label}. {text}" for label, text in zip(data_dict['label'], data_dict['text'])]
            formatted_choices = " ".join(combined_list)
        except (SyntaxError, ValueError):
            formatted_choices = choices
        input = f"{sample['question']} \nOptions: \n{formatted_choices}"
    elif dataset_type == 'race':
        input = f"{sample['question']} Based on the following article:\n{sample['article']}\nOptions: \n{sample['options']}"
    else:
        raise Exception("Dataset type not recognized.")
    prompt = f"Question: [{input}] Please answer the following multiple-choice question and only give me the selected option and provide your confidence level. \
    Note that the confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. Make sure you answer in the following structure: \n \
    [Answer]: , \n[Confidence (0-100)]: \n \
    Note: The confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. \
    For instance, if your confidence level is 80%, it means you are 80% certain that your answer is correct and there is a 20% chance that it may be incorrect. "
    prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:'''
    sample['text'] = prompt_template
    return sample
