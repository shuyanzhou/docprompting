import os

TQDM_DISABLE = True if 'TQDM_DISABLE' in os.environ and str(os.environ['TQDM_DISABLE']) == '1' else False
WANDB_DISABLE = True if 'WANDB_DISABLE' in os.environ and str(os.environ['WANDB_DISABLE']) == '1' else False

VAR_STR = "[[VAR]]"
NONE_STR = "[[NONE]]"
