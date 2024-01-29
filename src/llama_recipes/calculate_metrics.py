import pandas as pd
import numpy as np
import re


f = lambda x: [y+">" for y in x.split(">") if y]
def get_cat_list(x):
    y = f(x)
    y[-1] = "<" + y[-1].split("<")[1]
    y = y[:-2]
    return y

predictions = pd.read_csv("../predictions.csv")
predictions.dropna(inplace=True)
accuracy_per_cat_total = []
max_cat_level = max(len(x) for x in predictions["true"].apply(get_cat_list).tolist())
print("MAX CAT LEVEL: ", max_cat_level)
ground_truth = predictions["true"].to_numpy()
predictions = predictions["pred"].to_numpy()

for pred, true in zip(predictions, ground_truth):
    # clearing
    # print("BEFORE: ", pred)
    tags = re.findall(r'<(.*?)>', pred)
    numbers = re.search(r'\d+$', pred)
    pred = ''.join('<{}>'.format(tag) for tag in tags)# + numbers.group() if tags and numbers else ''
    
    # print("AFTER: ", pred)
    
    if pred:
        pred_cats = f(pred)
    else:
        pred_cats = " "
    # print("SPLITTED PRED: ", pred_cats)
    true = get_cat_list(true) 
    # print("SPLITTED TRUE: ", true)
    #print("GT: ", true)
    # print("==========================")
    len_true = len(true)
    accuracy_per_cat = [0] * max_cat_level
    counter = 0
    for pred_cat, true_cat in zip(pred_cats[:len_true], true):
        if pred_cat == true_cat:
            accuracy_per_cat[counter] = 1
            counter += 1
        else:
            break
    accuracy_per_cat_total.append(accuracy_per_cat)

final_accuracy_per_cat = np.array(accuracy_per_cat_total).sum(axis=0) / ground_truth.shape[0]
print("Final accuracy per cat: ", final_accuracy_per_cat)