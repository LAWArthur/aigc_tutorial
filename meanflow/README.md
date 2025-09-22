# meanflow

An empirical study of one-step generation technology for self-developed chip imaging business.

## Train
```Shell
# for debug
torchrun --standalone --nproc_per_node=1 --master_port=12345 train.py --dataset cifar10 --img_size 32 --model dhw_net --batch_size 8 --lr 0.0006 --save_ckpt_freq 10 --epochs 16000 --compute_fid --log_per_step 10 --tr_sampler v0 --P_mean_t -2.0 --P_std_t 2.0 --P_mean_r -2.0 --P_std_r 2.0 --warmup_epochs 200  --norm_p 0.75 --ratio 0.75 --dropout 0.2 --use_edm_aug

# train on CIFAR10-v0-CFG (bs 128x8)
torchrun --standalone --nproc_per_node=8 --master_port=12345 train.py --dataset cifar10 --img_size 32 --model dhw_net --batch_size 1024 --lr 0.0001 --save_ckpt_freq 100 --epochs 16000 --compute_fid --log_per_step 50 --tr_sampler v0 --P_mean_t -2.0 --P_std_t 2.0 --P_mean_r -2.0 --P_std_r 2.0 --warmup_epochs 200  --norm_p 0.75 --ratio 0.75 --dropout 0.2 --use_edm_aug

# train on CIFAR10-v0 (bs 128x8)
torchrun --standalone --nproc_per_node=8 --master_port=12345 train.py --dataset cifar10 --img_size 32 --model dhw_net --batch_size 1024 --lr 0.0001 --save_ckpt_freq 100 --epochs 16000 --compute_fid --log_per_step 50 --tr_sampler v0 --P_mean_t -2.0 --P_std_t 2.0 --P_mean_r -2.0 --P_std_r 2.0 --warmup_epochs 200  --norm_p 0.75 --ratio 0.75 --dropout 0.2 --use_edm_aug --class_cond

# train on CIFAR10-v1

```