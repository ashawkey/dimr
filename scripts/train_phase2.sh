export PATH="/usr/local/cuda-9.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.1/lib64:${LD_LIBRARY_PATH}"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train.py --config config/rfs_phase2_scannet.yaml
