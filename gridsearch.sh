CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data cnn -shot 0 >> cnn_0shot.log
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data cnn -shot 1 >> cnn_1shot.log
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data cnn -shot 3 >> cnn_3shot.log
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data cnn -shot 5 >> cnn_5shot.log
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data xsum -shot 0 >> xsum_0shot.log
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data xsum -shot 1 >> xsum_1shot.log
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data xsum -shot 3 >> xsum_3shot.log
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 asym_accept_dist.py --seed 0 --data xsum -shot 5 >> xsum_5shot.log
 




