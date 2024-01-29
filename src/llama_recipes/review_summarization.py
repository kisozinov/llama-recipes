from llama_recipes.inference.model_utils import load_model
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
import torch
import os
import sys
import json
from tqdm import tqdm
#from pymongo import MongoClient
from typing import List
# client = MongoClient()
# client = MongoClient("10.11.10.118")
# db = client.amazon
# collection = db.metadata
import csv

seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False, # if train_config.quantization else None,
    device_map="auto", # if train_config.quantization else None,
)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {
        
        "pad_token": "<PAD>",
    }
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt = """
You are a review summarizer system. You will get multiple reviews of the same product in Amazon electronics in input. Extract only benefits of the product and enumerate them. Do not generate anything before and after list. Your job is to give out the benefits of all the reviews at once, not each one individually
"""
def format_sample(sample, tokenizer):
    content = B_SYS + system_prompt + E_SYS
    for idx, review in enumerate(sample["reviewText"]):
        content += f"\nReview #{idx}\n{review}"
    sample = {
            "role": "user",
            "content": content
        }
    dialog_tokens = tokenizer.encode(
                f"{B_INST} {(sample['content']).strip()} {E_INST}",
            ) + [tokenizer.eos_token_id]
    return dialog_tokens


#chats = format_tokens(dialogs, tokenizer)
#print("CHATS: ", chats)

data_filename = "../raw_reviews_electronic.json"
with open(data_filename, 'r') as file:
    data = json.load(file)

summaries = {}
#buffer = {}
output_filename = "../summaries_l1.json"

with torch.no_grad():
    for idx, sample in enumerate(tqdm(data)):
        # if idx >= 300:
        #     break
        if idx % 10 == 0:
            with open(output_filename, "w") as file:
                json.dump(summaries, file)
        if len(sample["reviewText"]) < 3 and sample["asin"] not in summaries:
            summaries[sample["asin"]] = sample["reviewText"]
        else:
            chat = format_sample(sample, tokenizer)
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda")
            outputs = model.generate(
                input_ids=tokens,
                max_new_tokens=256,
                do_sample=False,
                top_p=1.0,
                #temperature=0.5,
                use_cache=True,
                #top_k=25,
                repetition_penalty=1,
                length_penalty=1,
            )

            output_text: str = tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(f"ASIN: {sample['asin']}")
            print(f"Model output:\n{output_text.split('[/INST]')[1]}")
            if sample["asin"] not in summaries:
                summaries[sample["asin"]] = []
            summaries[sample["asin"]].append(output_text)

with open(output_filename, "w") as file:
    json.dump(summaries, file)