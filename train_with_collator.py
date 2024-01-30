from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, default_data_collator, \
    DataCollatorWithPadding, get_linear_schedule_with_warmup, DataCollatorForTokenClassification,GPT2LMHeadModel,Seq2SeqTrainer, \
DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from peft import get_peft_config, get_peft_model, PromptTuningInit, LoraConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_dataset, collator_fn
from transformers import DataCollatorForLanguageModeling



device = "cuda" if torch.cuda.is_available() else 'cpu'
model_name_or_path = r'E:\pythonWork\nlp\train_new_gpt2\tmp\test-clm-sp' if device == 'cpu' else '/home/chenjq/pythonWork/nlp/train_new_gpt2/tmp/test-clm-sp-v4/checkpoint-4000'



tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# peft_model_id = "/home/chenjq/pythonWork/nlp/train_text_generation/output_with_collator-v2/checkpoint-10940"
# peft_config = LoraConfig.from_pretrained(peft_model_id)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32, lora_alpha=16, lora_dropout=0.3, bias="all",
)

model = get_peft_model(model, peft_config)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForTokenClassification(tokenizer)
data_collator2 = DataCollatorWithPadding(tokenizer)
data_collator3 = DataCollatorForLanguageModeling(tokenizer, mlm=False)
data_collator4 = DataCollatorForSeq2Seq(tokenizer)



all_data = get_dataset(tokenizer, device)
split_dataset = all_data['train'].train_test_split(test_size=0.3, seed=123)


model = model.to(device)

training_args = TrainingArguments(
    output_dir="./output_with_collator-v3",
    evaluation_strategy="epoch",
    learning_rate=3e-2,
    weight_decay=0.01,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    save_strategy="epoch",
    logging_steps=100,
    save_total_limit=5,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator4,
)

trainer.train()