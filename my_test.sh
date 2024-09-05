module load conda
conda activate pytorch_py39
CUDA_VISIBLE_DEVICES=0,1,2 python test_py.py
