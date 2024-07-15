#/!bin/bash
export PYTHONPATH=./:$PYTHONPATH

model=$1
num_gpus=$2
use_port=2688

train_batch_size=80
test_batch_size=150

seed=1028
opt=adam
lr=1e-4

warmup_lr=1e-4
warmup_epochs=5

decay_epochs=3
decay_rate=0.2
sched=step
proto_epochs=8
epochs=18
output_dir=output_cosine/
input_size=224
dim=64

# Loss
features_lr=$lr
add_on_layers_lr=3e-3
prototype_vectors_lr=3e-3
add_on_layers_final_lr=5e-4
prototype_vectors_final_lr=5e-4

use_ortho_loss=True
ortho_coe=1e-4
attri_coe=0.50
mse_coe=0.50
consis_coe=0.50
consis_thresh=0.20
cls_dis_coe=1.0
sep_dis_coe=0.20

use_mse_loss=true

ft=train

for data_set in CUB2011;
do
    prototype_num=2000
    
    python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=$use_port --use_env main.py \
        --seed=$seed \
        --output_dir=$output_dir/$data_set/$model/$seed-$lr-$opt-$epochs-$ft \
        --data_set=$data_set \
        --data_path=$data_path \
        --train_batch_size=$train_batch_size \
        --test_batch_size=$test_batch_size \
        --base_architecture=$model \
        --input_size=$input_size \
        --prototype_shape $prototype_num $dim 1 1 \
        --use_ortho_loss=$use_ortho_loss \
        --ortho_coe=$ortho_coe \
        --attri_coe=$attri_coe \
        --mse_coe=$mse_coe \
        --consis_coe=$consis_coe \
        --consis_thresh=$consis_thresh \
        --cls_dis_coe=$cls_dis_coe \
        --sep_dis_coe=$sep_dis_coe \
        --opt=$opt \
        --sched=$sched \
        --lr=$lr \
        --features_lr=$features_lr \
        --add_on_layers_lr=$add_on_layers_lr \
        --prototype_vectors_lr=$prototype_vectors_lr \
        --add_on_layers_final_lr=$add_on_layers_final_lr \
        --prototype_vectors_final_lr=$prototype_vectors_final_lr \
        --epochs=$epochs \
        --warmup_epochs=$warmup_epochs \
        --proto_epochs=$proto_epochs \
        --decay_epochs=$decay_epochs \
        --decay_rate=$decay_rate \
        --use_mse_loss=$use_mse_loss
done