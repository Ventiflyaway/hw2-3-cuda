# 1.performance and implementation

## perfromance before implementation
### theoretical complexity
- By the saying in the assignment description, if we compute the forces on the particles 
by iterating through every pair of them, we would have time O(n^2).
- In this assignment, we hope to achieve O(n) interactions.

### original implementation
every thread is responsible for 1 particle, 
every particle computes its ax ay with all particles one by one.

The following is the running time before any modification.
```bash
-n <int>: set number of particles
-o <filename>: set the output file name
-s <int>: set particle initialization seed
Simulation Time = 0.732337 seconds for 1000 particles.
Simulation Time = 1.19536 seconds for 2000 particles.
Simulation Time = 2.29949 seconds for 4000 particles.
Simulation Time = 5.62793 seconds for 10000 particles.
```

## After implementation
### theoretical analysis
I decided to implement Spatial Binning: giving a bounding box of size cutoff, 
only compute the particles inside that cutoff.

Description of the data structures: Arrays r used to acheieve this goal.
- cell_counts: number of particles in a cell
- cell_offset: records the first particle position in a cell
  -  (use Prefix Sum to compute this offset)
- cell_particles: exact particle in a cell
- cell_write_counts: record the # of particles insert into one cell, avoid race conditions

The implementation details are: 
- 1.sent each paticle to a thread on GPU,compute their corresponding cell_ids,
increment the cell_count[cell_id]
- 2.compute the starting index(cell_offset) according to cell_count
- 3.add paticles into cell_particles using cell_write_counts

### results
- A plot in the log-log scale showing your parallel codes performance 



# 2.Description of the synchronization I used in the GPU implementation.

# 3.Description of the design choices that I tried and how did they affect the performance.

# 4. runtime analysis
- break down the runtime into computation time, synchronization time and/or communication time 
- how do they scale with the number of particles?


------------------------------------------------------------------
# environment configuration in WSL Ubuntu:
```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc
source ~/.bashrc
# check if the CUDA Toolkit is successfully installed
nvcc --version
ls /usr/local/cuda-13.0/include/cuda.h
```
press left CTRL + left click on mouse to jump to a definition

------------------------------------------------------------------
记录关于GPU的知识：
# 一、Thread 负责计算，Warp 负责硬件执行，Block 负责组织协作与调度，Grid 代表整个 Kernel 的线程集合
# 二、block的作用
## 1. scheduling
- GPU 以 Block 为单位，将线程分配到不同的 Streaming Multiprocessors（SM）上执行。
- 一个 Block 的线程在同一个 SM 上运行，但不一定同时执行
## 2. shared mem
- 同一个 Block 内的线程可以利用高速 Shared Memory 交换数据，减少访问较慢的 Global Memory
## 3. synchronization
- 同一个 Block 内的线程可以使用：__syncthreads()在一个同步点等待同一个 Block 中的其他线程

# 三、warp
Warp 是 GPU 实际执行和调度指令的重要单位，同一 Warp 的线程以 SIMT 方式执行

# 四、有关GPU的代码
- __device__: function runs on GPU and can only be called from the GPU
- __global__: function runs on GPU and can be called from the CPU
- GPU Kernel cannot read CPU global variables