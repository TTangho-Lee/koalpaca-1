from datasets import load_dataset
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

import torch


########################

data_files = {
    "train": "train_koalpaca.jsonl",
    "validation": "valid_koalpaca.jsonl"
    }
data = load_dataset("json", data_files=data_files)
data = data.map(
    lambda x: {'text': f"### 질문: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" }
)

#################

#model_id = "EleutherAI/polyglot-ko-12.8b"
model_id = "Beomi/KoAlpaca-Polyglot-5.8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)

#############

data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

################

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

##############

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

#################

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h"
    ], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

##################

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        num_train_epochs=10,
        logging_steps=10,
        learning_rate=1e-4,
        output_dir="outputs",
        save_strategy="epoch"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

