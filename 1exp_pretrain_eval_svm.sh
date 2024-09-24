### posbert model train on shapenet, test on Modelnet40_strl
export CUDA_VISIBLE_DEVICES=0

test_name=pretrain
python -m torch.distributed.launch \
       --nproc_per_node=1 \
       ./eval_svm.py \
       --pointcloud_size 2048 \
       --arch pct_large \
       --pretrained_weights outputs_3D/$test_name/checkpoint.pth \
       --batch_size_per_gpu 16 \
       --exp_name $test_name \
       --dump_features outputs_3D/$test_name/svm_features