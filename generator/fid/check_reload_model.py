import sys
from fid_to_reload import change_name
import os
# if it is a reload checkpoint, check its existence, if not exist, convert from its original saved results
def main():
    args = sys.argv[1]
    print(args)
    model_name = None
    for idx, tok in enumerate(args.split()):
        if tok == '--model_name':
            model_name = args.split()[idx+1]
            break
    assert model_name

    if 'checkpoint' in model_name:
        assert '.reload' in model_name
        path = model_name.replace(".reload", "")
        if not os.path.exists(f"{model_name}/pytorch_model.bin"):
            change_name(path, model_name)
    else:
        print("good to go")

if __name__ == "__main__":
    main()