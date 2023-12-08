import argparse
import yaml
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
from utils import _get_max_length, load_model, preprocess_dataset
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, DataCollatorWithPadding, Trainer, TrainingArguments

# logging.set_verbosity(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter("always")

#create a list of args to pass to the script
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_name", type=str, default="lmsys/vicuna-7b-v1.5", help="model name")
parser.add_argument("--dataset_name", type=str, help="dataset name")
parser.add_argument("--output_dir", type=str, default="output", help="output directory")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("--warmup_steps", type=int, default=5, help="warmup steps")
parser.add_argument("--logging_steps", type=int, default=1, help="logging steps")
parser.add_argument("--eval_strategy", type=str, default="epoch", help="evaluation strategy")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument('--cot', action='store_true', help='sing chain of thought training')
parser.add_argument("--train_size", type=int, help="number of training examples")
parser.add_argument("--evaluation", action='store_true' help="whether evaluate the model after each epoch")
parser.add_argument("--skip_example", action='store_true' help="whether to skip the example output after training")

def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def train(model, tokenizer, dataset, output_dir, trainargs, train_size, evaluation):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = pad_token_id
    
    if train_size is not None:
        dataset['train'] = dataset['train'].select(range(train_size))
    # Training parameters
    if evaluation is True:
        trainer = Trainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset = dataset['validation'],
            args = trainargs,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
    else:
        trainargs.evaluation_strategy = None
        trainer = Trainer(
            model=model,
            train_dataset=dataset['train'],
            args = trainargs,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    # Launch training
    print("Training...")
    train_result = trainer.train()
    print("Training done!")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)

    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    trainer.push_to_hub()
    print(f"model successfully pushed to hub. username/{output_dir}")
    
    # # Free memory for merging weights
    # del model
    # del trainer
    # torch.cuda.empty_cache()

    return trainer

def example_output(dataset, tokenizer, model):
    model.config.use_cache = True
    print("-"*50)
    print("Example output from the finetuned model.")
    text = dataset['validation'][0]['text']
    input_text = text[:text.find("<INST/>")+7]
    print(f"input text: {input_text}")
    inputs=tokenizer.encode(input_text, return_tensors='pt').to('cuda')
    outputs = model.generate(inputs=inputs, max_length=1000, num_return_sequences=1)
    print(f"generated text:")
    for i, output in enumerate(outputs):
        print(f"{i}: {tokenizer.decode(output)}")

def main():
    args, args_text = _parse_args()
    print(args_text)

    #get all the args from the parser
    base_model_name = args.base_model_name
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_train_epochs = args.epochs
    warmup_steps = args.warmup_steps
    logging_steps = args.logging_steps
    eval_strategy = args.eval_strategy
    seed = args.seed
    cot = args.cot
    train_size = args.train_size
    evaluation = args.evaluation

    # Save args to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        f.write(args_text)

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    dataset = load_dataset(dataset_name)
    model, tokenizer = load_model(base_model_name, bnb_config)
    max_length = _get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, seed, dataset, cot, dataset_name)
    trainargs=TrainingArguments(
            per_device_train_batch_size=batch_size,
            evaluation_strategy = eval_strategy,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            output_dir=output_dir,
            optim="paged_adamw_8bit",
            num_train_epochs=num_train_epochs,
            push_to_hub = True
    )

    trainer = train(model, tokenizer, dataset, output_dir, trainargs, train_size, evaluation)
    if args.skip_example is None:
        example_output(dataset, tokenizer, model)

if __name__ == "__main__":
    main()

    



