# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 16
block_size = 512
gradient_accumulation_steps = 2 * 128 #per_gpu x num_gpus

# model
n_layer = 24
n_head = 32
n_embd = 4096
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 30 # total number of training iterations

# axonn params
G_intra_d=16
G_intra_c=1
G_intra_r=1
compile=False # disable compile for axonn
gradient_checkpointing=True

# this makes total number of tokens be 300B
max_iters = 30
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 1
log_interval = 10

# weight decay
weight_decay = 1e-1

# log every iteration
log_interval=1