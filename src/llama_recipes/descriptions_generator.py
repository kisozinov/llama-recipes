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
#from pymongo import MongoClient
from typing import List
# client = MongoClient()
# client = MongoClient("10.11.10.118")
# db = client.amazon
# collection = db.metadata


seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
model_name = "meta-llama/Llama-2-13b-chat-hf"
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
Give five descriptions of product below in natural language, based on product information in JSON format.
Do not try to make an advertisement, use narrative style.
The "feature" field contains important features of the product that you need to mention in the descriptions you receive.
The "reviewText" field is a list of reviews for this product, extract the main benefits of the product from them.
Output result should be a list with generated descriptions.
Don't generate anything unnecessary before and after descriptions!
You must generate in natural language.
"""
# Example of one description: 1. Asus Zenfone 2 ZE551ML 4/32GB Black (90AZ00A1-M01470) smartphone is a device that was released by Asus in 2015. \\
# It is a 4G LTE enabled model that allows users to access the internet quickly and conveniently. The smartphone is equipped with a \\
# screen with a resolution of 1920x1080 pixels and a diagonal of 5.5 inches, which uses IPS technology. Thanks to this, users \\
# can enjoy a bright and clear image. The device is equipped with a 4-core processor, 4GB of RAM and 32GB of internal memory to ensure high performance and ample storage.\\
# The smartphone is also equipped with a 13-megapixel camera capable of taking quality pictures.
# Try to use different product characteristics in different descriptions. \\
def format_sample(sample, tokenizer):
    sample = {
            "role": "user",
            "content": B_SYS
            + system_prompt
            + E_SYS
            + str(sample),
        }
    dialog_tokens = tokenizer.encode(
                f"{B_INST} {(sample['content']).strip()} {E_INST}",
            ) + [tokenizer.eos_token_id]
    return dialog_tokens


#chats = format_tokens(dialogs, tokenizer)
#print("CHATS: ", chats)

data_filename = "../preprocessed_electronics.json"
with open(data_filename, 'r') as file:
    data = json.load(file)

with torch.no_grad():
    for sample in data:
        chat = format_sample(sample, tokenizer)
        tokens= torch.tensor(chat).long()
        print("TOTAL TOKENS: ", tokens.shape)
        tokens= tokens.unsqueeze(0)
        tokens= tokens.to("cuda")
        outputs = model.generate(
            input_ids=tokens,
            max_new_tokens=2048,
            do_sample=True,
            top_p=1.0,
            temperature=0.5,
            use_cache=True,
            top_k=25,
            repetition_penalty=1,
            length_penalty=1,
        )

        output_text: str = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Model output:\n{output_text.split(E_INST, maxsplit=1)[1]}")
        print("\n==================================\n")
