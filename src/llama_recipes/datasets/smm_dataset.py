# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools
import pandas as pd


# def tokenize_dialog(dialog, tokenizer):
#     prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
#     answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
#     dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
#     #Add labels, convert prompt token to -100 in order to ignore in loss function
#     labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

#     combined_tokens = {
#         "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
#         "labels": list(itertools.chain(*(t for t in labels_tokens))),
#     }

#     return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):
    df = pd.read_parquet(f"../smm_descriptions_1201/{split}.parquet")
    print(df.head(5))
    dataset = datasets.Dataset.from_pandas(df, split=split)

    prompt = (
        f"""
        Предскажи по описанию товара на Мегамаркете его идентификатор, \\
        состоящий из следа по дереву категорий, и ID в конце. Вот описание: \n{{description}}\n
        """
    )

    def apply_prompt_template(sample):
        return {"description": prompt.format(description=sample["description"]),
                "item_id": sample["item_id"]
        }
    
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    #special_tokens = {"additional_special_tokens": ["[ID]", "[/ID]"]}
    

    def tokenize_add_label(sample):
        # print(tokenizer.tokenize(sample["item_id"]))
        # exit()
        description = tokenizer.encode(tokenizer.bos_token + sample["description"], add_special_tokens=True)
        response = tokenizer.encode(sample["item_id"] + tokenizer.eos_token, add_special_tokens=True)
        #item_id = tokenizer.encode(sample["item_id"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": description + response,
            "attention_mask" : [1] * (len(description) + len(response)),
            "labels": [-100] * len(description) + response,
            }

        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    #print(dataset[:5])
    #tokenizer.add_tokens(['[ID]','[/ID]'])
    print(dataset)

    return dataset
