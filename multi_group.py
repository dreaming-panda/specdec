import torch.distributed as dist
import torch
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

group = dist.new_group(backend="gloo")
DEVICE = torch.device("cuda", local_rank)

cuda_id = torch.LongTensor([local_rank]).to(DEVICE).repeat(100) + local_rank
cuda_id_cpu = torch.LongTensor([local_rank]).repeat(100) + local_rank
dist.all_reduce(cuda_id, dist.ReduceOp.SUM)
dist.all_reduce(cuda_id_cpu, dist.ReduceOp.SUM, group=group)
if local_rank == 0:
    print(cuda_id)
    print(cuda_id_cpu)

