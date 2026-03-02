# AllReduce-Your-Expectations

> **Manual Ring All-Reduce vs. Native Collective Synchronization on LeNet-5**

This repository contains a high-performance implementation of **Distributed Data Parallelism (DDP)** from scratch. We train a LeNet-5 CNN on MNIST using 4 distributed CPU processes, comparing the industry-standard `dist.all_reduce` against a custom-built **Ring All-Reduce** algorithm.

---

## 📝 Project Description
Distributed training is often bottlenecked by communication, not computation. When multiple CPU cores train a model, they must share gradients to stay synchronized. 

This project solves the "Communication Bottleneck" by implementing a **Ring-based topology**. Instead of a "Master-Worker" setup where one node is overwhelmed, each node in our "Ring" only talks to its immediate neighbor. By using **Gradient Packing**, we concatenate all model parameters into a single buffer, reducing the overhead of dozens of small network calls into one efficient circular handoff.

---

## 🛠️ The Ring All-Reduce Algorithm
The manual implementation breaks the gradient synchronization into two mathematically rigorous phases:



1.  **Scatter-Reduce Phase**: Each of the $N$ ranks partitions its local gradient into $N$ chunks. Chunks are passed around the circle $N-1$ times. In each step, a rank receives a chunk from its left neighbor and sums it with its local data.
2.  **All-Gather Phase**: The fully summed chunks are then circulated through the ring an additional $N-1$ times so that every rank ends up with the complete, globally averaged gradient vector.



---

## 🧠 Mathematical Formulation
In both implementations, we ensure that every rank updates its local parameters $\theta$ using the global average gradient $\bar{g}$:

$$\bar{g} = \frac{1}{N} \sum_{i=0}^{N-1} \nabla L_i(\theta)$$

The Ring All-Reduce algorithm is bandwidth-optimal. The total data transferred per node is:
$$\text{Data Sent} = 2 \frac{N-1}{N} S$$
where $S$ is the total number of parameters in the model. This makes the communication cost nearly independent of the number of nodes $N$.

---

## 📊 Performance Benchmarks
*Tested on: Wells Fargo OptiPlex-5090 (4 CPU Ranks)*

| Method | Total Time | Test Accuracy | Improvement |
| :--- | :--- | :--- | :--- |
| **Native All-Reduce** | 34.02s | 77.50% | Baseline |
| **Manual Ring All-Reduce** | **26.87s** | **77.50%** | **~21% Faster** |

### **Key Takeaway**
The identical accuracy (**77.50%**) across both methods proves that our manual Ring logic is mathematically equivalent to the native PyTorch/Gloo collectives. The speedup is purely an architectural win—by "packing" gradients and using a circular buffer, we minimized the latency ($\alpha$) that usually slows down distributed training.

---

## 🚀 How to Run
1. **Clone the repo**:
   ```bash
   git clone [https://github.com/your-username/One-Ring-To-Reduce-Them-All.git](https://github.com/your-username/One-Ring-To-Reduce-Them-All.git)
   cd One-Ring-To-Reduce-Them-All
