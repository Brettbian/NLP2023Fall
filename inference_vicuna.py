import argparse
import bitsandbytes as bnb
from datasets import load_dataset
import os
import torch
import random
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, DataCollatorWithPadding, Trainer, TrainingArguments
from utils import _get_max_length, preprocess_dataset
import csv
from tqdm import tqdm

# logging.set_verbosity(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter("always")

#create a list of args to pass to the script
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="lmsys/vicuna-7b-v1.5", help="model name")
parser.add_argument("--dataset_name", type=str, help="dataset name")
parser.add_argument("--output_dir", type=str, default="output", help="output directory")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--cot", type = bool, help = "using chain of thought training")



def main():
    dataset = load_dataset("brettbbb/vicuna_qa_causal_LM_split",split = "test")
    new_model = "brettbbb/vicuna_mc_finetune"
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()
    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ## Preprocess dataset
    max_length = _get_max_length(model)
    seed = 1

    dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
    dataset.set_format("torch")

    # Specify the file name
    file_name = "result.csv"

    # Open the CSV file in write mode
    with open(file_name, mode='w', newline='') as file:
        # Define the CSV writer
        writer = csv.writer(file)

        # Write header
        writer.writerow(['input_text', 'answer', 'generated_output'])

        for i in tqdm(range(len(dataset))):
            answer = dataset[i]['answer']
            input_text = dataset[i]['formatted_prompt']
            inputs=tokenizer.encode(input_text, return_tensors='pt').to('cuda')
            outputs = model.generate(inputs=inputs, max_length=1000, num_return_sequences=1)
            decoded_outputs = [tokenizer.decode(output) for output in outputs]

            # Write the data for each iteration
            writer.writerow([input_text, answer, decoded_outputs])

    print(f"Data has been written to {file_name}")




