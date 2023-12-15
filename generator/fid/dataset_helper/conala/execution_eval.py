"""
execution-based evaluation
"""
import argparse
import json
import sys
import evaluate
from datasets import load_metric
import numpy as np
from collections import defaultdict
import os

from py import test 
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def pass_at_k(result_file, unittest_file):
    with open(unittest_file, 'r') as f:
        unittests = json.load(f)
    
    # select the examples which have unit test
    selected_predictions = []
    with open(result_file, 'r') as f:
        for line in f:
            pred = json.loads(line)
            if pred["question_id"] in unittests:
                selected_predictions.append(pred)
    print(f"selected {len(selected_predictions)} examples with unit test")
                
    # run the test   
    # load the metric from huggingface 
    code_eval_metric = evaluate.load("code_eval")
    preds = []
    tests = []
    for prediction in selected_predictions:
        suffix = ""
        question_id = prediction["question_id"]
        unittest = unittests[question_id]
        entry_point = unittest["entry_point"]
        test_func = f"\n{unittest['test']}\ncheck({entry_point})"
        
        # wrap the generated code to a runnable function
        if isinstance(prediction['clean_code'], list):
            runnable_func = [f"{unittest['prompt']}{x}{suffix}" for x in prediction['clean_code']]
        else:
            runnable_func = [f"{unittest['prompt']}{prediction['clean_code']}{suffix}"]

        preds.append(runnable_func)
        tests.append(test_func)
        
    r = code_eval_metric.compute(
        predictions=preds,
        references=tests,
        k=[1, 5, 10, 50, 100, 150, 200],
        num_workers=8,
    )
    print(r[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="")
    args = parser.parse_args()
    result_file = args.result_file
    unittest_file = "data/conala/unittest_docprompting_conala.json"
    assert result_file
    pass_at_k(result_file, unittest_file)
