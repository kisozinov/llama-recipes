from llama_recipes.inference.model_utils import load_peft_model
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
import torch
import os
import sys
import pandas as pd
from tqdm import tqdm

seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
model_name = "meta-llama/Llama-2-7b-chat-hf"
prompt_file = "../amazon_electronics_val_v3.parquet"
pretrained_model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False, # if train_config.quantization else None,
    # device_map="auto", # if train_config.quantization else None,
)
#model = pretrained_model
model = load_peft_model(pretrained_model, "/home/jovyan/kisozinov/saved_models/llama2_amazon_v3")
tokenizer = LlamaTokenizer.from_pretrained(model_name)
with open("../categories.txt", "r") as f:
    extra_tokens = []
    for l in f.readlines():
        extra_tokens.append(l[:-1])
extra_tokens.append("<SOI>")
extra_tokens.append("<EOI>")
print("extra tokens: ", extra_tokens)
special_tokens_dict = {'additional_special_tokens': extra_tokens}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)#, special_tokens=True)
print("We have added", num_added_tokens, "tokens")
print("LEN OF TOKENIZER: ", len(tokenizer))

model.resize_token_embeddings(len(tokenizer))
tokenizer.add_special_tokens(
    {
        
        "pad_token": "<PAD>",
    }
)
#print("We have added", num_added_tokens, "tokens")

#model.resize_token_embeddings(len(tokenizer))

data = pd.read_parquet(prompt_file)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt = """
You are a system that gives the product identifier based on the product description by user. Your answer should look like: 'The product identifier is: ...'
"""
prompt = (
    f"""
    Below I give you information about the product on amazon:
    Title:\n{{title}}\n
    Brand:\n{{brand}}\n
    Description:\n{{description}}\n
    Feature:\n{{feature}}\n
    Price:\n{{price}}\n
    What identifier of this product?
    """
)
def format_row(row, tokenizer):
    user_prompt = prompt.format(
                    title=row["title"],
                    brand=row["brand"],
                    description=row["description"],
                    feature=row["feature"],
                    price=row["price"]
    )
    sample = {
        "role": "user",
        "content": B_SYS
        + system_prompt
        + E_SYS
        + user_prompt,
    }
    dialog_tokens = tokenizer.encode(
                f"{B_INST} {(sample['content']).strip()} {E_INST}",
            ) + [tokenizer.eos_token_id]
    return dialog_tokens
    # return f"{B_INST} {(sample['content']).strip()} {E_INST}"

# chats = format_tokens(dialogs, tokenizer)
#print("CHATS: ", chats)
# chats = []
# batch_size = 4
predictions = {"pred": [], "true": []}
# count = 0
with torch.no_grad():
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        # count += 1
        # if count == 100:
        #     break
        chat = format_row(row, tokenizer)
        tokens= torch.tensor(chat).long()
        tokens= tokens.unsqueeze(0)
        tokens= tokens.to("cuda:0")
    # for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
    #     chats.append(format_row(row, tokenizer))
        # if (idx + 1) % batch_size != 0:
            #print(chats)
            # batch = tokenizer(chats, padding='max_length', truncation=True, max_length=50, return_tensors="pt")
            # chats = []
            # batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model.generate(
            # **batch,
            input_ids=tokens,
            max_new_tokens=32,
            do_sample=True,
            top_p=1.0,
            temperature=0.8,
            use_cache=True,
            min_length=5,
            top_k=50,
            repetition_penalty=1,
            length_penalty=1,
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        final_output = output_text.split(E_INST, maxsplit=1)[1].replace("<s>", "").replace("</s>", "")
        predictions["pred"].append(final_output)
        predictions["true"].append(row["target"])
        print(f"Model output:\n{output_text}")
        print("\n==================================\n")

pred_df = pd.DataFrame.from_dict(predictions)
pred_df.to_csv("../predictions.csv")