from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from functools import partial
import os

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
    DATASET_TYPE = _identify_dataset_type(dataset_name)
    INSTRUCTION_KEY = "<s>[INST]"
    INSTRUCTION_END = "</INST>"
    END_OF_SENTENCE = "</s>"
    if DATASET_TYPE == 'mctest':
        #replace //newline with \n
        sample['Story'] = sample['Story'].replace('\\newline', '')
        prompt = f"Question: {sample['Question']} Based on the following article:\n{sample['Story']}.\nOptions: \n{sample['Options']}."
        predicted_answer = sample['Predicted Answer']
        actual_answer = sample['Actual Answer']
    if DATASET_TYPE == 'race':
        prompt = f"Question: {sample['question']} Based on the following article:\n{sample['article']}.\nOptions: \n{sample['options']}."
        predicted_answer = sample['predicted_answer']
        actual_answer = sample['answer']
    if DATASET_TYPE =='commonsenseqa':
        prompt = f"Question: {sample['question']}.\nOptions: \n{sample['choices']}."
        predicted_answer = sample['predicted_answer']
        actual_answer = sample['answerKey']
    if DATASET_TYPE == 'arc':
        prompt = f"Question: {sample['question']}.\nOptions: \n{sample['choices']}."
        predicted_answer = sample['predicted_answer']
        actual_answer = sample['answerKey']
    else:
        raise Exception("Dataset type not recognized.")
    if cot:
        if predicted_answer == actual_answer:
            sample['text'] = f"{INSTRUCTION_KEY}{prompt}{INSTRUCTION_END} Explanation: {sample['explanation']}{END_OF_SENTENCE}"
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
    print(dataset["train"][0])

    return dataset

