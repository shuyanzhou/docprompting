"""
Convert SimCSE's checkpoints to Huggingface style.
"""

import argparse
import shutil

import torch
import os
import json

def change_name(path, new_path=None):
    if new_path is None:
        new_path = path
    state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=torch.device("cpu"))
    new_state_dict = {}
    keep = []
    change = []
    for key, param in state_dict.items():
        if key.startswith("encoder.encoder"):
            key = key.replace("encoder.encoder", "encoder")
            key = key.replace("module.layer", "layer")
            change.append(key)
        else:
            keep.append(key)
        new_state_dict[key] = param

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    torch.save(new_state_dict, os.path.join(new_path, "pytorch_model.bin"))
    print(f"kept keys: {keep}")
    print(f"changed keys: {change}")
    for file in os.listdir(path):
        if file != 'pytorch_model.bin':
            shutil.copyfile(os.path.join(path, file), os.path.join(new_path, file))

    for name in ['config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'tokenizer_config.json']:
        shutil.copyfile(os.path.join('/home/shuyanzh/workshop/op_agent/data/fid/codet5-base', name), os.path.join(new_path, name))
    print("Copy tokenization files from codet5-base folder to the target folder")

    # Change architectures in config. json
    # config = json.load(open(os.path.join(path, "config.json")))
    # for i in range(len(config["architectures"])):
    #     config["architectures"][i] = config["architectures"][i].replace("ForCL", "Model")
    # json.dump(config, open(os.path.join(new_path, "config.json"), "w"), indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path of SimCSE checkpoint folder")
    parser.add_argument('--new_path', type=str, help='New place to save checkpoints', default=None)
    args = parser.parse_args()
    args.path = '/home/shuyanzh/workshop/op_agent/data/fid/code_t5_nothing/checkpoint/best_dev'
    args.new_path = '/home/shuyanzh/workshop/op_agent/data/fid/code_t5_nothing/checkpoint/best_dev.reload'
    print("FiD checkpoint -> Fid Reload checkpoint for {}".format(args.path))
    change_name(args.path, args.new_path)


if __name__ == "__main__":
    main()
