from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

import torch
checkpoint_path = "outputs/checkpoint-3110"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, quantization_config=bnb_config, device_map={"":0})
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.eval()
model.config.use_cache = True

def gen(x):
    inputs = tokenizer(
        f"### 질문: {x}\n\n### 답변:", 
        return_tensors='pt', 
        return_token_type_ids=False
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    gened = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0], skip_special_tokens=True))


gen("다음 두 문장 중에서 올바른 문장을 선택하고, 그 이유를 설명하세요.\n1. 히치하이크를 하다.\n2. 힛치하이크를 하다.")