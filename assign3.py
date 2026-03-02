import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import datasets, transforms
import os
import time
import numpy as np


# 1. LeNet-5

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 2. Manual Ring All-Reduce

def ring_allreduce(tensor, world_size, rank):
    original_size = tensor.numel()

    # Pad tensor if needed
    if original_size % world_size != 0:
        pad_size = world_size - (original_size % world_size)
        tensor = torch.cat([tensor, torch.zeros(pad_size)])
    else:
        pad_size = 0

    chunk_size = tensor.numel() // world_size
    chunks = list(torch.chunk(tensor, world_size))

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size

    # ---- Phase 1: Scatter-Reduce ----
    for i in range(world_size - 1):
        send_idx = (rank - i) % world_size
        recv_idx = (rank - i - 1) % world_size

        send_buf = chunks[send_idx].clone()
        recv_buf = torch.zeros_like(chunks[recv_idx])

        send_req = dist.isend(send_buf, right)
        recv_req = dist.irecv(recv_buf, left)

        send_req.wait()
        recv_req.wait()

        chunks[recv_idx] += recv_buf

    # ---- Phase 2: All-Gather ----
    for i in range(world_size - 1):
        send_idx = (rank - i + 1) % world_size
        recv_idx = (rank - i) % world_size

        send_buf = chunks[send_idx].clone()
        recv_req = dist.irecv(chunks[recv_idx], left)
        send_req = dist.isend(send_buf, right)

        send_req.wait()
        recv_req.wait()

    result = torch.cat(chunks)
    return result[:original_size]


# 3. Training Worker

def train_worker(rank, world_size, method):

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29510'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(42)
    np.random.seed(42)


   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST('./data', train=True, download=(rank==0), transform=transform)

    indices = np.random.choice(len(full_train), 100, replace=False)
    subset = Subset(full_train, indices)

    sampler = DistributedSampler(
        subset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(subset, batch_size=25, sampler=sampler)

    # Test loader 
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256)

   
    # Model
    
    model = LeNet5()

    # Broadcasting
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

   
    # Training
   
    dist.barrier()
    start_time = time.time()

    for epoch in range(100):

        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        # Accumulate gradients over local epoch
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

        # ---- Gradient Sync AFTER EPOCH ----
        if method == "allreduce":
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

        elif method == "ring":
            flat_grads = torch.cat([p.grad.view(-1) for p in model.parameters()])
            reduced = ring_allreduce(flat_grads, world_size, rank)
            reduced /= world_size

            offset = 0
            for p in model.parameters():
                numel = p.grad.numel()
                p.grad.copy_(reduced[offset:offset+numel].view_as(p.grad))
                offset += numel

        optimizer.step()

    dist.barrier()
    total_time = time.time() - start_time

    
    # Evaluation 
   
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = torch.tensor([correct, total], dtype=torch.float32)
    dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    final_accuracy = (acc[0] / acc[1]) * 100

    if rank == 0:
        print(f"Method: {method:10} | Time: {total_time:6.2f}s | Test Accuracy: {final_accuracy.item():.2f}%")

    dist.destroy_process_group()


# 4. Main

if __name__ == "__main__":
    world_size = 4

    for method in ["allreduce", "ring"]:
        print(f"\n--- Running {method.upper()} ---")
        mp.spawn(train_worker, args=(world_size, method), nprocs=world_size, join=True)

