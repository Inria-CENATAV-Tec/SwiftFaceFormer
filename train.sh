module load conda
conda activate pytorch_py39
export OMP_NUM_THREADS=3
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train/train_arc.py
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
