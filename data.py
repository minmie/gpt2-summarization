import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else 'cpu'

def get_dataset(tokenizer, device):
    path = r'E:\pythonWork\datasets\LCSTS_new\LCSTS_new\dev.json' if device=='cpu' else "/home/chenjq/datasets/LCSTS_new/train.json"

    raw_dataset = load_dataset("json", data_files=path)
    # raw_dataset['train'] = raw_dataset['train'].select(range(10000))


    def process(examples):

        sample_input_ids = tokenizer.encode('<bos><unused0>'+examples["content"]+'<unused1>', add_special_tokens=False)
        label_input_ids = tokenizer.encode(examples["summary"], add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids = sample_input_ids + label_input_ids
        labels = [-100] * len(sample_input_ids) + label_input_ids
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(labels) == len(attention_mask)

        return {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "labels": labels,
        }

    split_data = raw_dataset.map(process,
                              remove_columns=raw_dataset["train"].column_names,
                              # load_from_cache_file=False
                              )

    print(1)
    return split_data

def collator_fn(batch):
    batch_data = [each.values() for each in batch]
    # input_ids, attention_mask, labels= zip(*batch_data)
    input_ids, labels= zip(*batch_data)
    input_ids = [torch.tensor(i, dtype=torch.long) for i in input_ids]
    # attention_mask = [torch.tensor(i, dtype=torch.long) for i in attention_mask]
    labels = [torch.tensor(i, dtype=torch.long) for i in labels]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=2).to(device)
    # attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0).to(device)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
    return {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "labels": labels,
        }




if __name__ == '__main__':
    model_name_or_path = r'E:\pythonWork\nlp\train_new_gpt2\tmp\test-clm-sp' if device == 'cpu' else '/home/chenjq/pythonWork/nlp/train_new_gpt2/tmp/test-clm-sp-v4/checkpoint-4000'

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    get_dataset(tokenizer, device)