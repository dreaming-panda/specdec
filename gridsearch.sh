CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data cnn --shot 0 >> Tcnn_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data cnn --shot 1 >> Tcnn_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data cnn --shot 3 >> Tcnn_3shot.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 sirius_accept.py --seed 0 --data xsum --shot 0 >> Xxsum_0shot.log
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 sirius_accept.py --seed 0 --data xsum --shot 1 >> Xxsum_1shot.log
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=8 sirius_accept.py --seed 0 --data xsum --shot 3 >> Xxsum_3shot.log



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data gsm8k --shot 0 >> Tgsm8k_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data gsm8k --shot 1 >> Tgsm8k_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data gsm8k --shot 3 >> Tgsm8k_3shot.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data orca_math --shot 0 >> Torca_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data orca_math --shot 1 >> Torca_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data orca_math --shot 3 >> Torca_3shot.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data python --shot 0 >> Tpython_0shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data python --shot 1 >> Tpython_1shot.log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 sirius_accept.py --seed 0 --data python --shot 3 >> Tpython_3shot.log







