import argparse
from datasets import load_dataset
import os
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import _get_max_length, get_answer_column, generate_prompt
import csv
from tqdm import tqdm

# logging.set_verbosity(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter("always")

#create a list of args to pass to the script
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_name", type=str, default="lmsys/vicuna-7b-v1.5", help="model name")
parser.add_argument("--dataset_name", type=str, help="dataset name")
parser.add_argument("--finetuned_model", type=str, help="new model name")
parser.add_argument("--output_dir", type=str, default="output", help="output directory")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--split", type = str, default="validation",help = "split of the dataset to use")

def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = _parse_args()
    print(args_text)
    #get all the args from the parser
    base_model_name = args.base_model_name
    new_model_name = args.finetuned_model
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    seed = args.seed
    split = args.split

    # Save args to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "inference_args.txt"), "w") as f:
        f.write(args_text)

    dataset = load_dataset(dataset_name,split = split)
    new_model = new_model_name
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ## Preprocess dataset
    max_length = _get_max_length(model)
    seed = 1

    dataset = dataset.map(generate_prompt, batched=True, fn_kwargs={"dataset_name": dataset_name})
    dataset.set_format("torch")

    print(dataset[0])

    # Specify the file name
    file_name = "inference_result.csv"

    output_path = os.path.join(output_dir, file_name)

    answer_column = get_answer_column(dataset_name)

    print("Start Inferencing...")
    # Open the CSV file in write mode
    with open(output_path, mode='w', newline='') as file:
        # Define the CSV writer
        writer = csv.writer(file)

        # Write header
        writer.writerow(['input_text', 'answer', 'generated_output'])

        for i in tqdm(range(len(dataset))):
            answer = dataset[i][answer_column]
            prompt = dataset[i]['text']
            inputs = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
            outputs = model.generate(inputs=inputs, max_length=1000, num_return_sequences=1)
            decoded_outputs = [tokenizer.decode(output) for output in outputs]

            # Write the data for each iteration
            writer.writerow([prompt, answer, decoded_outputs])

            # Check if it's a multiple of 10 and update the CSV file
            if (i + 1) % 10 == 0:
                file.flush()

    print(f"Data has been written to {output_path}")




