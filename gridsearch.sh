CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data cnn --shot 0 >> Zcnn_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data cnn --shot 1 >> Zcnn_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data cnn --shot 3 >> Zcnn_3shot.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 griffin_accept_dist.py --seed 0 --data xsum --shot 0 >> Xxsum_0shot.log
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 griffin_accept_dist.py --seed 0 --data xsum --shot 1 >> Xxsum_1shot.log
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 griffin_accept_dist.py --seed 0 --data xsum --shot 3 >> Xxsum_3shot.log



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data gsm8k --shot 0 >> Zgsm8k_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data gsm8k --shot 1 >> Zgsm8k_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data gsm8k --shot 3 >> Zgsm8k_3shot.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data orca_math --shot 0 >> Zorca_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data orca_math --shot 1 >> Zorca_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data orca_math --shot 3 >> Zorca_3shot.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data python --shot 0 >> Zpython_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data python --shot 1 >> Zpython_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=4 griffin_accept_dist.py --seed 0 --data python --shot 3 >> Zpython_3shot.log







