from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, default_data_collator, \
    DataCollatorWithPadding, get_linear_schedule_with_warmup,GPT2LMHeadModel
from peft import get_peft_config, get_peft_model, PromptTuningInit, LoraConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_dataset, collator_fn
from transformers import DataCollatorForLanguageModeling



device = "cuda" if torch.cuda.is_available() else 'cpu'
model_name_or_path = r'E:\pythonWork\nlp\train_new_gpt2\tmp\test-clm-sp' if device == 'cpu' else '/home/chenjq/pythonWork/nlp/train_new_gpt2/tmp/test-clm-sp-v5/checkpoint-4000'



tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=32, lora_alpha=16, lora_dropout=0.3, bias="all",
# )
#
# model = get_peft_model(model, peft_config)

tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorWithPadding(tokenizer)




all_data = get_dataset(tokenizer, device)
split_dataset = all_data['train'].train_test_split(test_size=0.3)


model = model.to(device)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f'num of train samples:{len(train_dataset)}')
print(f'num of eval samples:{len(eval_dataset)}')
batch_size = 32

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collator_fn, batch_size=batch_size, pin_memory=False
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=collator_fn, batch_size=batch_size, pin_memory=False)
num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
global_step = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        global_step+=1
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if global_step%100==0:
            print(f'epoch={epoch}/{num_epochs},step={global_step}, loss={loss}, lr={lr_scheduler.get_last_lr()}')

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    path = f'./output_model/ckpt-{global_step}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
