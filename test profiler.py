import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

model = torch.nn.Linear(10, 10).cuda()
x = torch.randn(5, 10).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
    on_trace_ready=tensorboard_trace_handler('./log/trace_new'),
    record_shapes=True,
    profile_memory=True
) as prof:
    for _ in range(4):
        y = model(x)
        prof.step()
## tensorboard --logdir=./log
