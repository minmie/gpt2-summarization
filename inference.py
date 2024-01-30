import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

device='cuda'
# base_model = '/home/chenjq/pythonWork/nlp/train_new_gpt2/tmp/test-clm-sp-v5/checkpoint-4000'
base_model1 = '/home/chenjq/pythonWork/nlp/train_text_generation/output_model/ckpt-160870'
# peft_model_id = "/home/chenjq/pythonWork/nlp/train_text_generation/output_model/ckpt-32174"

# config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model1)
model = AutoModelForCausalLM.from_pretrained(base_model1)
# model = PeftModel.from_pretrained(model1, peft_model_id)

# text = "<bos><unused0>{}<unused1>".format( "步入深水区的房地产调控政策走向，再度引发官媒聚焦。15日，新华社旗下《经济参考报》报道称，相关内部会议透露，将加快研究包括土地、金融、财税等方面的房地产中长期调控政策。“去行政化”将成为未来调控方向。")
text = "<bos><unused0>{}<unused1>".format( "63岁退休教师谢淑华，拉着人力板车，历时1年，走了2万4千里路，带着年过九旬的妈妈环游中国，完成了妈妈“一辈子在锅台边转，也想出去走走”的心愿。她说：“妈妈愿意出去走走，我就愿意拉着，孝心不能等，能走多远就走多远。")
# text = "<bos><unused0>{}<unused1>".format( "昨天，上海女子教育联盟和上海开放大学女子学院成立揭牌。上海开放大学女子学院国顺路、中山西路院区以及首批试点的闵行区、闸北区和长宁区学习中心，都已推出首期学习菜单。上海市妇联有关人士表示，男性也可以报名参加学习。")

inputs = tokenizer.encode(
    text, return_tensors="pt", add_special_tokens=False
)

model.to('cuda')
# tokenizer.decode()


with torch.no_grad():
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs.to(device),
        # input_ids=inputs["input_ids"],
        # attention_mask=inputs["attention_mask"],
        num_beams=10,
        max_new_tokens=30,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        # max_length=50,
        top_p=0.8,
        top_k=6,
        temperature=0.7
    )
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=False))
