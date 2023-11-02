"""prepare data for captions
"""

import json
import pandas as pd
import numpy as np
import collections
import re
from sklearn.metrics import accuracy_score


def clean_evidence(evidences):
    output = []
    for evidence in evidences:
        evidence = re.sub(' +', ' ', evidence)
        evidence = evidence.replace("[Image Caption]: ", "")
        evidence = evidence.replace("[Image Alternative Text]: ", "")
        evidence = evidence.replace("[Title of the news that the given image occurred]: ", "")
        evidence = evidence.strip()
        output.append(evidence)
    return ' '.join(output)

with open("/home/zehong/FastChat/fastchat/llm_judge/data/captions/inverse_caps_balanced_test_4229_fullymatched.json", "r") as ff:
    data = json.load(ff)

defined_prompt = "You will be provided with a caption and a text. You only need to output whether a 'caption' is true (relevant) or false (irrelevant) given a 'text'."

for k, v in data.items():
    cur_q = {}
    cur_q["question_id"] = v['idx']
    cur_q["category"] = "reasoning"
    
    cleaned_evidence = clean_evidence(v["inverse_evidence"])
    
    text = defined_prompt + "\n caption: " + v["cap1"] + "\n text: " + cleaned_evidence + "\n output: "
    cur_q["turns"] = []
    cur_q["turns"].append(text)
    cur_q["turns"].append("Explain your reasons.")
    
    cur_q["reference"] = [v["idx"].split('-')[-1]]   
    
    with open("/home/zehong/Fastchat/llm_judge/data/captions/question.jsonl", "a") as fout:
        fout.write(json.dumps(cur_q) + "\n")