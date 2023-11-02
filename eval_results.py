import json
import pandas as pd
import numpy as np
import collections
from sklearn.metrics import accuracy_score

check_questions = []
with open("/home/zehong/Fastchat/llm_judge/data/captions/question.jsonl", "r") as ques_file:
    for line in ques_file:
        if line:
            check_questions.append(json.loads(line))
            
check_answers = []
with open("/home/zehong/Fastchat/llm_judge/data/captions/model_answer/vicuna-13b-v1.5.jsonl", "r") as ans_file:
    for line in ans_file:
        if line:
            check_answers.append(json.loads(line))

output_preds = []
gt, preds = [], []
for ans in check_answers:
    temp = {}
    cur_id = ans["question_id"]
    temp["question_id"] = ans["question_id"]
    
    for que in check_questions:
        if cur_id == que["question_id"]: 
            temp["text"] = que["turns"][0]
            temp["ground_truth"] = que["reference"][0]
            gt.append(1 if que["reference"][0]== "real" else 0)
            
    get_ans = ans['choices'][0]["turns"][0].lower()
    if 'true' in get_ans:
        get_ans = "True"
    else:
        get_ans = "False"
    temp["prediction"] = get_ans
    preds.append(1 if get_ans=="True" else 0)
    temp["reason"] = ans['choices'][0]["turns"][1] 
    temp["model_id"] = ans["model_id"]
    output_preds.append(temp)
output_df = pd.DataFrame(output_preds)
print(accuracy_score(gt, preds))
output_df.to_csv("/home/zehong/Fastchat/llm_judge/preds-vicuna-13b.csv")