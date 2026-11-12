#include "common.h"
#include <cuda.h>
#include <cub/cub.cuh>

#define NUM_THREADS 256 // every block has 256 threads

// Put any static global variables here that you will use throughout the simulation.
int blks;
int num_cells_per_row;
int num_cells;

int* cell_counts;// # of particles in each cell
int* cell_offsets;// starting index of each cell in the cell_particles array
int* cell_particles;// particle indices in each cell
int* cell_write_counts;// # of particles written to each cell in the cell_particles array

void* scan_temp = nullptr;
size_t scan_bytes = 0;

// & means we directly change the particle in the GPU memory,
// instead of copying it to the CPU and changing it there
// __device__: function runs on GPU and can only be called from the GPU
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;// distance squared between the two particles

    // if the distance > cutoff distance, do not compute
    if (r2 > cutoff * cutoff)
        return;
    
    // r2 = fmax( r2, min_r*min_r );  ensure distance is not too small to avoid numerical instability
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx; // accumulate
    particle.ay += coef * dy;
}

// __global__: function runs on GPU and can be called from the CPU
__global__ void compute_cell_gpu(particle_t* particles, int num_parts, int num_cells_per_row, int* cell_counts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) // if the thread ID is larger than the number of particles, return
        return;

    // compute which cell the particle is in, add particle to that cell
    int cx = int(particles[tid].x / cutoff);
    int cy = int(particles[tid].y / cutoff);
    
    // Handle boundary
    cx = min(cx, num_cells_per_row - 1);
    cy = min(cy, num_cells_per_row - 1);

    int cell_id = cx + cy * num_cells_per_row;
    atomicAdd(&cell_counts[cell_id], 1);
}

void compute_offsets_gpu() {
    cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes,
        cell_counts, cell_offsets, num_cells);
}

__global__ void scatter_particles_gpu(particle_t* particles, int num_parts, int num_cells_per_row, int* cell_counts, int* cell_offsets, int* cell_particles, int* cell_write_counts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) // if the thread ID is larger than the number of particles, return
        return;

    // add particle to cell_particles array
    int cx = int(particles[tid].x / cutoff);
    int cy = int(particles[tid].y / cutoff);
    // Handle boundary
    cx = min(cx, num_cells_per_row - 1);
    cy = min(cy, num_cells_per_row - 1);

    int cell_id = cx + cy * num_cells_per_row;

    // get the starting index of the cell in the cell_particles array and add the particle index to that cell
    int slot = atomicAdd(&cell_write_counts[cell_id], 1);
    int offset = cell_offsets[cell_id] + slot;

    cell_particles[offset] = tid;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int num_cells_per_row, int* cell_counts, int* cell_offsets, int* cell_particles, int* cell_write_counts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) // if the thread ID is larger than the number of particles, return
        return;

    particles[tid].ax = particles[tid].ay = 0; // reset acceleration to 0 before computing forces

    // compute which cell the particle is in
    int cx = int(particles[tid].x / cutoff);
    int cy = int(particles[tid].y / cutoff);
    // Handle boundary
    cx = min(cx, num_cells_per_row - 1);
    cy = min(cy, num_cells_per_row - 1);

    int cell_id = cx + cy * num_cells_per_row;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
             
            if (nx < 0 || nx >= num_cells_per_row || ny < 0 || ny >= num_cells_per_row) {
                continue; // skip out-of-bounds neighbor cells
            }
            int neighbor_cell_id = nx + ny * num_cells_per_row;
            for (int i = 0; i < cell_counts[neighbor_cell_id]; i++) {
                int neighbor_id = cell_particles[cell_offsets[neighbor_cell_id] + i];

                apply_force_gpu(particles[tid], particles[neighbor_id]);
            }
        }
    }
}


__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

// compute block numbers according to the number of particles and threads/block
void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Grid configuration
    num_cells_per_row = int(size / cutoff) + 1;
    num_cells = num_cells_per_row * num_cells_per_row;

    // Allocate GPU memory
    cudaMalloc(&cell_counts, num_cells * sizeof(int));
    cudaMalloc(&cell_offsets, (num_cells + 1) * sizeof(int));
    cudaMalloc(&cell_particles, num_parts * sizeof(int));
    cudaMalloc(&cell_write_counts, num_cells * sizeof(int));

    cudaMemset(cell_counts, 0, num_cells * sizeof(int));

    // Compute the size of temporary storage needed for the exclusive scan -> bytes
    cub::DeviceScan::ExclusiveSum( nullptr, scan_bytes,
    cell_counts, cell_offsets, num_cells);

    // Allocate once
    cudaMalloc(&scan_temp, scan_bytes);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Compute forces
    // CUDA kernel launch: kernel<<<number_of_blocks, threads_per_block>>>(arguments);

    cudaMemset(cell_counts, 0, num_cells * sizeof(int));
    cudaMemset(cell_write_counts, 0, num_cells * sizeof(int));

    //each particle computes which cell it belongs to and increments the count of that cell
    compute_cell_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, num_cells_per_row, cell_counts);
    // compute the starting index of each cell in the cell_particles array
    compute_offsets_gpu();
    // scatter the particles into the cell_particles array according to their cell
    scatter_particles_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, num_cells_per_row, cell_counts, cell_offsets, cell_particles, cell_write_counts);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    compute_forces_gpu<<<blks, NUM_THREADS>>>(
    parts, num_parts, num_cells_per_row,
    cell_counts, cell_offsets, cell_particles, cell_write_counts
    );

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, end);

    printf("Compute Forces: %.4f ms\n", elapsed_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    // // compute the forces on each particle by looking at its neighboring cells
    // compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, num_cells_per_row, cell_counts, cell_offsets, cell_particles, cell_write_counts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
