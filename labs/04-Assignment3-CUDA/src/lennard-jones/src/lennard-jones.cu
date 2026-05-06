#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <utility>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#include "gifenc.h"
#include "lennard-jones.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#define PARTICLE_LIMIT 4000
#define NEIGHBOR_COUNT 9

#define R_CUT2 ((double)(R_CUT) * (double)(R_CUT))
#define SIGMA2 ((double)(SIGMA) * (double)(SIGMA))
#define FOUR_EPSILON (4.0 * (double)(EPSILON))
#define TWENTYFOUR_EPSILON (24.0 * (double)(EPSILON))
#define HALF_DT (0.5 * (double)(DT))

#define LJ_SR ((double)(SIGMA) / (double)(R_CUT))
#define LJ_SR2 (LJ_SR * LJ_SR)
#define LJ_SR6 (LJ_SR2 * LJ_SR2 * LJ_SR2)
#define LJ_SR12 (LJ_SR6 * LJ_SR6)
#define V_SHIFT (FOUR_EPSILON * (LJ_SR12 - LJ_SR6))

__constant__ constants d_c;

static inline constants make_constants(double box_size, float r_skin, unsigned int n) {
    constants c;

    c.box_size = box_size;
    c.inv_box_size = 1.0 / box_size;

    c.max_disp2 = 0.25 * (double)r_skin * (double)r_skin;

    const double target_cell_size = (double)R_CUT + (double)r_skin;

    c.num_cells_x = (int)(box_size / target_cell_size);
    c.num_cells_y = (int)(box_size / target_cell_size);

    if (c.num_cells_x < 3) c.num_cells_x = 3;
    if (c.num_cells_y < 3) c.num_cells_y = 3;

    c.num_cells = c.num_cells_x * c.num_cells_y;

    c.cell_size_x = box_size / (double)c.num_cells_x;
    c.cell_size_y = box_size / (double)c.num_cells_y;

    c.inv_cell_size_x = 1.0 / c.cell_size_x;
    c.inv_cell_size_y = 1.0 / c.cell_size_y;

    c.max_neighbors = (int)n - 1;

    return c;
}

#if GENERATE_GIF
uint8_t palette[] = {
    0, 0, 0,
    255, 255, 0
};

void set_pixel(uint8_t *img, int w, int h, int x, int y) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }

    const size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = 1;
}

void render_frame_gif(
    ge_GIF *gif,
    const double *__restrict__ x,
    const double *__restrict__ y,
    unsigned int n,
    double box_size
) {
    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);

    const double frame_x_scale = (double)(FRAME_WIDTH - 1) / box_size;
    const double frame_y_scale = (double)(FRAME_HEIGHT - 1) / box_size;

    const int frame_height_minus_1 = FRAME_HEIGHT - 1;
    const int radius = FRAME_PARTICLE_RADIUS;
    const int radius2 = radius * radius;

    for (unsigned int i = 0; i < n; ++i) {
        const int px = (int)(x[i] * frame_x_scale);
        int py = (int)(y[i] * frame_y_scale);

        py = frame_height_minus_1 - py;

        for (int dy = -radius; dy <= radius; ++dy) {
            const int dy2 = dy * dy;

            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy2 <= radius2) {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy);
                }
            }
        }
    }
}
#endif

double random_double(void) {
    static const double inv_rand_max = 1.0 / (double)RAND_MAX;
    return (double)rand() * inv_rand_max;
}

double compute_v_shift(void) {
    return V_SHIFT;
}

double compute_ke(const double *vx, const double *vy, unsigned int n) {
    double ke = 0.0;

    for (unsigned int i = 0; i < n; ++i) {
        ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
    }

    return ke;
}

int initialize_particles(
    double *__restrict__ x,
    double *__restrict__ y,
    double *__restrict__ vx,
    double *__restrict__ vy,
    double *__restrict__ fx,
    double *__restrict__ fy,
    unsigned int n,
    double box_size,
    double placement_fraction,
    unsigned int seed,
    double temperature
) {
    srand(seed);

    const double inv_n = 1.0 / (double)n;

    const unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    const double inv_n_side = 1.0 / (double)n_side;

    const double placement_size = placement_fraction * box_size;
    const double offset = 0.5 * (box_size - placement_size);
    const double delta = placement_size * inv_n_side;
    const double jitter_delta = JITTER * delta;

    double mean_vx = 0.0;
    double mean_vy = 0.0;

    for (unsigned int k = 0; k < n; k++) {
        const double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        const double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        x[k] = x0 + (2.0 * random_double() - 1.0) * jitter_delta;
        y[k] = y0 + (2.0 * random_double() - 1.0) * jitter_delta;

        vx[k] = 2.0 * random_double() - 1.0;
        vy[k] = 2.0 * random_double() - 1.0;

        fx[k] = 0.0;
        fy[k] = 0.0;

        mean_vx += vx[k];
        mean_vy += vy[k];
    }

    mean_vx *= inv_n;
    mean_vy *= inv_n;

    double ke = 0.0;

    for (unsigned int k = 0; k < n; k++) {
        vx[k] -= mean_vx;
        vy[k] -= mean_vy;

        ke += 0.5 * (
            vx[k] * vx[k] +
            vy[k] * vy[k]
        );
    }

    const double current_temperature = ke * inv_n;

    if (current_temperature <= 0.0) {
        return 0;
    }

    const double scale = sqrt(temperature / current_temperature);

    for (unsigned int k = 0; k < n; k++) {
        vx[k] *= scale;
        vy[k] *= scale;
    }

    return 1;
}

__global__ void compute_ke_partial_kernel(
    const double *__restrict__ vx,
    const double *__restrict__ vy,
    unsigned int n,
    double *__restrict__ partials
) {
    extern __shared__ double shared_ke[];

    const unsigned int tid = threadIdx.x;
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    double local_ke = 0.0;

    for (unsigned int i = global_id; i < n; i += stride) {
        local_ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
    }

    shared_ke[tid] = local_ke;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_ke[tid] += shared_ke[tid + offset];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared_ke[0];
    }
}

static double compute_ke_cuda(
    const double *__restrict__ d_vx_arr,
    const double *__restrict__ d_vy_arr,
    unsigned int n,
    double *__restrict__ d_ke_partial,
    int num_ke_blocks,
    int ke_block_size
) {
    compute_ke_partial_kernel<<<
        num_ke_blocks,
        ke_block_size,
        ke_block_size * sizeof(double)
    >>>(
        d_vx_arr,
        d_vy_arr,
        n,
        d_ke_partial
    );
    checkCudaErrors(cudaGetLastError());

    thrust::device_ptr<double> d_ke_ptr(d_ke_partial);

    return thrust::reduce(
        d_ke_ptr,
        d_ke_ptr + num_ke_blocks,
        0.0,
        thrust::plus<double>()
    );
}

__global__ void count_cells_kernel(
    const double *__restrict__ x,
    const double *__restrict__ y,
    unsigned int n,
    int *__restrict__ particle_cell_indices,
    int *__restrict__ cell_start
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    int cell_x = (int)(x[i] * d_c.inv_cell_size_x);
    int cell_y = (int)(y[i] * d_c.inv_cell_size_y);

    if (cell_x >= d_c.num_cells_x) cell_x = d_c.num_cells_x - 1;
    if (cell_y >= d_c.num_cells_y) cell_y = d_c.num_cells_y - 1;
    if (cell_x < 0) cell_x = 0;
    if (cell_y < 0) cell_y = 0;

    const int cell = cell_y * d_c.num_cells_x + cell_x;

    particle_cell_indices[i] = cell;

    atomicAdd(&cell_start[cell + 1], 1);
}

__global__ void scatter_particles_kernel(
    const double *__restrict__ x,
    const double *__restrict__ y,
    const double *__restrict__ vx,
    const double *__restrict__ vy,
    unsigned int n,
    const int *__restrict__ particle_cell_indices,
    int *__restrict__ next_pos,
    double *__restrict__ temp_x,
    double *__restrict__ temp_y,
    double *__restrict__ temp_vx,
    double *__restrict__ temp_vy,
    int *__restrict__ temp_particle_cell_indices
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const int cell = particle_cell_indices[i];
    const int pos = atomicAdd(&next_pos[cell], 1);

    temp_x[pos] = x[i];
    temp_y[pos] = y[i];
    temp_vx[pos] = vx[i];
    temp_vy[pos] = vy[i];

    temp_particle_cell_indices[pos] = cell;
}

__global__ void build_verlet_list_from_sorted_particles_kernel(
    const double *__restrict__ x,
    const double *__restrict__ y,
    const int *__restrict__ particle_cell_indices,
    const int *__restrict__ cell_start,
    const int *__restrict__ cell_neighbors,
    int *__restrict__ neighbor_counts,
    int *__restrict__ neighbor_list,
    unsigned int n,
    float r_skin
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const double local_box_size = d_c.box_size;
    const double inv_box_size = d_c.inv_box_size;

    const double r_list = (double)R_CUT + (double)r_skin;
    const double r_list2 = r_list * r_list;

    const int max_neighbors = d_c.max_neighbors;
    const int list_offset = (int)i * max_neighbors;

    const int cell_i = particle_cell_indices[i];

    const double xi = x[i];
    const double yi = y[i];

    int count = 0;

    for (int k = 0; k < NEIGHBOR_COUNT; ++k) {
        const int cell_j = cell_neighbors[cell_i * NEIGHBOR_COUNT + k];

        const int j_start = cell_start[cell_j];
        const int j_end = cell_start[cell_j + 1];

        for (int j = j_start; j < j_end; ++j) {
            if ((int)i == j) {
                continue;
            }

            double dx = xi - x[j];
            double dy = yi - y[j];

            dx -= local_box_size * nearbyint(dx * inv_box_size);
            dy -= local_box_size * nearbyint(dy * inv_box_size);

            const double dist2 = dx * dx + dy * dy;

            if (dist2 < r_list2) {
                neighbor_list[list_offset + count] = j;
                count++;
            }
        }
    }

    neighbor_counts[i] = count;
}

__global__ void compute_forces_verlet_kernel(
    const double *__restrict__ x,
    const double *__restrict__ y,
    double *__restrict__ fx,
    double *__restrict__ fy,
    unsigned int n,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_list,
    double *__restrict__ pe_partial
) {
    extern __shared__ double shared_pe[];

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double pe_local = 0.0;

    if (i < n) {
        const double local_box_size = d_c.box_size;
        const double inv_box_size = d_c.inv_box_size;

        const int max_neighbors = d_c.max_neighbors;
        const int count = neighbor_counts[i];
        const int list_offset = (int)i * max_neighbors;

        double fxi = 0.0;
        double fyi = 0.0;
        double pei = 0.0;

        const double xi = x[i];
        const double yi = y[i];

        for (int k = 0; k < count; ++k) {
            const int j = neighbor_list[list_offset + k];

            double dx = xi - x[j];
            double dy = yi - y[j];

            dx -= local_box_size * nearbyint(dx * inv_box_size);
            dy -= local_box_size * nearbyint(dy * inv_box_size);

            const double r2 = dx * dx + dy * dy;

            if (r2 >= R_CUT2 || r2 == 0.0) {
                continue;
            }

            const double inv_r2 = 1.0 / r2;
            const double sr2 = SIGMA2 * inv_r2;
            const double sr6 = sr2 * sr2 * sr2;
            const double sr12 = sr6 * sr6;

            const double fij =
                TWENTYFOUR_EPSILON * (2.0 * sr12 - sr6) * inv_r2;

            fxi += fij * dx;
            fyi += fij * dy;

            const double vij = FOUR_EPSILON * (sr12 - sr6) - V_SHIFT;

            pei += 0.5 * vij;
        }

        fx[i] = fxi;
        fy[i] = fyi;

        pe_local = pei;
    }

    shared_pe[tid] = pe_local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_pe[tid] += shared_pe[tid + offset];
        }

        __syncthreads();
    }

    if (tid == 0) {
        pe_partial[blockIdx.x] = shared_pe[0];
    }
}

__global__ void compute_forces_verlet_no_energy_kernel(
    const double *__restrict__ x,
    const double *__restrict__ y,
    double *__restrict__ fx,
    double *__restrict__ fy,
    unsigned int n,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_list
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const double local_box_size = d_c.box_size;
    const double inv_box_size = d_c.inv_box_size;

    const int max_neighbors = d_c.max_neighbors;
    const int count = neighbor_counts[i];
    const int list_offset = (int)i * max_neighbors;

    double fxi = 0.0;
    double fyi = 0.0;

    const double xi = x[i];
    const double yi = y[i];

    for (int k = 0; k < count; ++k) {
        const int j = neighbor_list[list_offset + k];

        double dx = xi - x[j];
        double dy = yi - y[j];

        dx -= local_box_size * nearbyint(dx * inv_box_size);
        dy -= local_box_size * nearbyint(dy * inv_box_size);

        const double r2 = dx * dx + dy * dy;

        if (r2 >= R_CUT2 || r2 == 0.0) {
            continue;
        }

        const double inv_r2 = 1.0 / r2;
        const double sr2 = SIGMA2 * inv_r2;
        const double sr6 = sr2 * sr2 * sr2;
        const double sr12 = sr6 * sr6;

        const double fij =
            TWENTYFOUR_EPSILON * (2.0 * sr12 - sr6) * inv_r2;

        fxi += fij * dx;
        fyi += fij * dy;
    }

    fx[i] = fxi;
    fy[i] = fyi;
}

__global__ void update_velocities_and_positions_kernel(
    double *__restrict__ x,
    double *__restrict__ y,
    double *__restrict__ vx,
    double *__restrict__ vy,
    const double *__restrict__ fx,
    const double *__restrict__ fy,
    unsigned int n,
    const double *__restrict__ temp_x,
    const double *__restrict__ temp_y,
    int *__restrict__ needs_rebuild
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const double local_box_size = d_c.box_size;
    const double inv_box_size = d_c.inv_box_size;
    const double max_disp2 = d_c.max_disp2;

    vx[i] += HALF_DT * fx[i];
    vy[i] += HALF_DT * fy[i];

    x[i] += DT * vx[i];
    y[i] += DT * vy[i];

    double wx = fmod(x[i], local_box_size);
    double wy = fmod(y[i], local_box_size);

    if (wx < 0.0) {
        wx += local_box_size;
    }

    if (wy < 0.0) {
        wy += local_box_size;
    }

    x[i] = wx;
    y[i] = wy;

    double dx = x[i] - temp_x[i];
    double dy = y[i] - temp_y[i];

    dx -= local_box_size * nearbyint(dx * inv_box_size);
    dy -= local_box_size * nearbyint(dy * inv_box_size);

    const double dist2 = dx * dx + dy * dy;

    if (dist2 > max_disp2) {
        atomicExch(needs_rebuild, 1);
    }
}

__global__ void update_velocities_second_half_kernel(
    double *__restrict__ vx,
    double *__restrict__ vy,
    const double *__restrict__ fx,
    const double *__restrict__ fy,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    vx[i] += HALF_DT * fx[i];
    vy[i] += HALF_DT * fy[i];
}

void build_cell_neighbors_full(
    int *__restrict__ cell_neighbors,
    const constants *__restrict__ c
) {
    const int num_cells = c->num_cells;
    const int num_cells_x = c->num_cells_x;
    const int num_cells_y = c->num_cells_y;

    for (int cell = 0; cell < num_cells; ++cell) {
        const int cx = cell % num_cells_x;
        const int cy = cell / num_cells_x;

        int k = 0;

        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                int nx = cx + ox;
                int ny = cy + oy;

                if (nx < 0) nx += num_cells_x;
                else if (nx >= num_cells_x) nx -= num_cells_x;

                if (ny < 0) ny += num_cells_y;
                else if (ny >= num_cells_y) ny -= num_cells_y;

                cell_neighbors[cell * NEIGHBOR_COUNT + k] =
                    ny * num_cells_x + nx;

                k++;
            }
        }
    }
}

void build_cell_neighbors(
    int *__restrict__ cell_neighbors,
    const constants *__restrict__ c
) {
    build_cell_neighbors_full(cell_neighbors, c);
}

void rebuild_cells_cuda(
    double *&d_x_arr,
    double *&d_y_arr,
    double *&d_vx_arr,
    double *&d_vy_arr,
    unsigned int n,
    int *&d_particle_cell_indices,
    int *__restrict__ d_particle_ids,
    int *__restrict__ d_cell_start,
    int *__restrict__ d_next_pos,
    double *&d_temp_x,
    double *&d_temp_y,
    double *&d_temp_vx,
    double *&d_temp_vy,
    int *&d_temp_particle_cell_indices,
    int num_cells
) {
    (void)d_particle_ids;

    const int block_size = 256;
    const int grid_size = (int)((n + block_size - 1) / block_size);

    checkCudaErrors(cudaMemset(
        d_cell_start,
        0,
        (size_t)(num_cells + 1) * sizeof(int)
    ));

    count_cells_kernel<<<grid_size, block_size>>>(
        d_x_arr,
        d_y_arr,
        n,
        d_particle_cell_indices,
        d_cell_start
    );
    checkCudaErrors(cudaGetLastError());

    thrust::device_ptr<int> start_ptr(d_cell_start);

    thrust::inclusive_scan(
        start_ptr,
        start_ptr + num_cells + 1,
        start_ptr
    );

    checkCudaErrors(cudaMemcpy(
        d_next_pos,
        d_cell_start,
        (size_t)num_cells * sizeof(int),
        cudaMemcpyDeviceToDevice
    ));

    scatter_particles_kernel<<<grid_size, block_size>>>(
        d_x_arr,
        d_y_arr,
        d_vx_arr,
        d_vy_arr,
        n,
        d_particle_cell_indices,
        d_next_pos,
        d_temp_x,
        d_temp_y,
        d_temp_vx,
        d_temp_vy,
        d_temp_particle_cell_indices
    );
    checkCudaErrors(cudaGetLastError());

    std::swap(d_x_arr, d_temp_x);
    std::swap(d_y_arr, d_temp_y);
    std::swap(d_vx_arr, d_temp_vx);
    std::swap(d_vy_arr, d_temp_vy);
    std::swap(d_particle_cell_indices, d_temp_particle_cell_indices);

    checkCudaErrors(cudaMemcpy(
        d_temp_x,
        d_x_arr,
        (size_t)n * sizeof(double),
        cudaMemcpyDeviceToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_temp_y,
        d_y_arr,
        (size_t)n * sizeof(double),
        cudaMemcpyDeviceToDevice
    ));
}

static void build_verlet_list_cuda(
    const double *__restrict__ d_x_arr,
    const double *__restrict__ d_y_arr,
    const int *__restrict__ d_particle_cell_indices,
    const int *__restrict__ d_cell_start,
    const int *__restrict__ d_cell_neighbors,
    int *__restrict__ d_neighbor_counts,
    int *__restrict__ d_neighbor_list,
    unsigned int n,
    float r_skin
) {
    const int block_size = 256;
    const int grid_size = (int)((n + block_size - 1) / block_size);

    build_verlet_list_from_sorted_particles_kernel<<<grid_size, block_size>>>(
        d_x_arr,
        d_y_arr,
        d_particle_cell_indices,
        d_cell_start,
        d_cell_neighbors,
        d_neighbor_counts,
        d_neighbor_list,
        n,
        r_skin
    );
    checkCudaErrors(cudaGetLastError());
}

double compute_forces_verlet_cuda(
    double *&d_x_arr,
    double *&d_y_arr,
    double *&d_vx_arr,
    double *&d_vy_arr,
    double *__restrict__ d_fx_arr,
    double *__restrict__ d_fy_arr,
    unsigned int n,
    int *&d_particle_cell_indices,
    int *__restrict__ d_particle_ids,
    int *__restrict__ d_cell_start,
    int *__restrict__ d_next_pos,
    double *&d_temp_x,
    double *&d_temp_y,
    double *&d_temp_vx,
    double *&d_temp_vy,
    int *&d_temp_particle_cell_indices,
    int *__restrict__ d_needs_rebuild,
    const int *__restrict__ d_cell_neighbors,
    int *__restrict__ d_neighbor_counts,
    int *__restrict__ d_neighbor_list,
    int num_cells,
    float r_skin,
    double *__restrict__ d_pe_partial,
    int force_blocks,
    int force_block_size,
    int compute_energy
) {
    int h_needs_rebuild = 0;

    checkCudaErrors(cudaMemcpy(
        &h_needs_rebuild,
        d_needs_rebuild,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (h_needs_rebuild) {
        rebuild_cells_cuda(
            d_x_arr,
            d_y_arr,
            d_vx_arr,
            d_vy_arr,
            n,
            d_particle_cell_indices,
            d_particle_ids,
            d_cell_start,
            d_next_pos,
            d_temp_x,
            d_temp_y,
            d_temp_vx,
            d_temp_vy,
            d_temp_particle_cell_indices,
            num_cells
        );

        build_verlet_list_cuda(
            d_x_arr,
            d_y_arr,
            d_particle_cell_indices,
            d_cell_start,
            d_cell_neighbors,
            d_neighbor_counts,
            d_neighbor_list,
            n,
            r_skin
        );

        h_needs_rebuild = 0;

        checkCudaErrors(cudaMemcpy(
            d_needs_rebuild,
            &h_needs_rebuild,
            sizeof(int),
            cudaMemcpyHostToDevice
        ));
    }

    if (compute_energy) {
        compute_forces_verlet_kernel<<<
            force_blocks,
            force_block_size,
            force_block_size * sizeof(double)
        >>>(
            d_x_arr,
            d_y_arr,
            d_fx_arr,
            d_fy_arr,
            n,
            d_neighbor_counts,
            d_neighbor_list,
            d_pe_partial
        );
        checkCudaErrors(cudaGetLastError());

        thrust::device_ptr<double> d_pe_ptr(d_pe_partial);

        return thrust::reduce(
            d_pe_ptr,
            d_pe_ptr + force_blocks,
            0.0,
            thrust::plus<double>()
        );
    }

    compute_forces_verlet_no_energy_kernel<<<
        force_blocks,
        force_block_size
    >>>(
        d_x_arr,
        d_y_arr,
        d_fx_arr,
        d_fy_arr,
        n,
        d_neighbor_counts,
        d_neighbor_list
    );
    checkCudaErrors(cudaGetLastError());

    return 0.0;
}

SimulationResult run_simulation(
    double *__restrict__ x,
    double *__restrict__ y,
    double *__restrict__ vx,
    double *__restrict__ vy,
    double *__restrict__ fx,
    double *__restrict__ fy,
    unsigned int n,
    unsigned int nsteps,
    double box_size,
    int log_steps,
    float r_skin,
    int block_size,
    double density
) {
    (void)density;

    SimulationResult out;

    out.n = n;
    out.start_kinetic = 0.0;
    out.start_potential = 0.0;
    out.start_total = 0.0;
    out.final_kinetic = 0.0;
    out.final_potential = 0.0;
    out.final_total = 0.0;

    const constants h_c = make_constants(box_size, r_skin, n);
    const int num_cells = h_c.num_cells;
    const int max_neighbors = h_c.max_neighbors;

    checkCudaErrors(cudaMemcpyToSymbol(
        d_c,
        &h_c,
        sizeof(constants)
    ));

    const int update_block_size = 256;
    const int update_grid_size = (int)((n + update_block_size - 1) / update_block_size);

    int force_block_size = block_size;

    if (force_block_size <= 0) {
        force_block_size = 256;
    }

    const int force_blocks = (int)((n + force_block_size - 1) / force_block_size);

    const int ke_block_size = 256;
    int num_ke_blocks = (int)((n + ke_block_size - 1) / ke_block_size);

    if (num_ke_blocks < 1) {
        num_ke_blocks = 1;
    }

    if (num_ke_blocks > 1024) {
        num_ke_blocks = 1024;
    }

    double *d_x_arr = NULL;
    double *d_y_arr = NULL;
    double *d_vx_arr = NULL;
    double *d_vy_arr = NULL;
    double *d_fx_arr = NULL;
    double *d_fy_arr = NULL;

    double *d_temp_x = NULL;
    double *d_temp_y = NULL;
    double *d_temp_vx = NULL;
    double *d_temp_vy = NULL;

    double *d_ke_partial = NULL;
    double *d_pe_partial = NULL;

    int *d_particle_cell_indices = NULL;
    int *d_particle_ids = NULL;
    int *d_cell_start = NULL;
    int *d_next_pos = NULL;
    int *d_temp_particle_cell_indices = NULL;
    int *d_needs_rebuild = NULL;
    int *d_cell_neighbors = NULL;

    int *d_neighbor_counts = NULL;
    int *d_neighbor_list = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_x_arr, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_y_arr, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_vx_arr, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_vy_arr, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_fx_arr, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_fy_arr, (size_t)n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&d_temp_x, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_y, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_vx, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_vy, (size_t)n * sizeof(double)));

    checkCudaErrors(cudaMalloc((void **)&d_particle_cell_indices, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_particle_ids, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_cell_start, (size_t)(num_cells + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_next_pos, (size_t)num_cells * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_particle_cell_indices, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_needs_rebuild, sizeof(int)));

    checkCudaErrors(cudaMalloc((void **)&d_neighbor_counts, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc(
        (void **)&d_neighbor_list,
        (size_t)n * (size_t)max_neighbors * sizeof(int)
    ));

    checkCudaErrors(cudaMalloc((void **)&d_ke_partial, (size_t)num_ke_blocks * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_pe_partial, (size_t)force_blocks * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_x_arr, x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y_arr, y, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vx_arr, vx, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vy_arr, vy, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_fx_arr, 0, (size_t)n * sizeof(double)));
    checkCudaErrors(cudaMemset(d_fy_arr, 0, (size_t)n * sizeof(double)));

    int *cell_neighbors = (int *)malloc(
        (size_t)num_cells * NEIGHBOR_COUNT * sizeof(int)
    );

    if (cell_neighbors == NULL) {
        fprintf(stderr, "Allocation failed for cell_neighbors\n");
        exit(EXIT_FAILURE);
    }

    build_cell_neighbors_full(cell_neighbors, &h_c);

    checkCudaErrors(cudaMalloc(
        (void **)&d_cell_neighbors,
        (size_t)num_cells * NEIGHBOR_COUNT * sizeof(int)
    ));

    checkCudaErrors(cudaMemcpy(
        d_cell_neighbors,
        cell_neighbors,
        (size_t)num_cells * NEIGHBOR_COUNT * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    free(cell_neighbors);
    cell_neighbors = NULL;

    int h_needs_rebuild = 1;

    checkCudaErrors(cudaMemcpy(
        d_needs_rebuild,
        &h_needs_rebuild,
        sizeof(int),
        cudaMemcpyHostToDevice
    ));

    out.start_potential = compute_forces_verlet_cuda(
        d_x_arr,
        d_y_arr,
        d_vx_arr,
        d_vy_arr,
        d_fx_arr,
        d_fy_arr,
        n,
        d_particle_cell_indices,
        d_particle_ids,
        d_cell_start,
        d_next_pos,
        d_temp_x,
        d_temp_y,
        d_temp_vx,
        d_temp_vy,
        d_temp_particle_cell_indices,
        d_needs_rebuild,
        d_cell_neighbors,
        d_neighbor_counts,
        d_neighbor_list,
        num_cells,
        r_skin,
        d_pe_partial,
        force_blocks,
        force_block_size,
        1
    );

    out.start_kinetic = compute_ke_cuda(
        d_vx_arr,
        d_vy_arr,
        n,
        d_ke_partial,
        num_ke_blocks,
        ke_block_size
    );

    out.start_total = out.start_kinetic + out.start_potential;

    out.final_potential = out.start_potential;
    out.final_kinetic = out.start_kinetic;
    out.final_total = out.start_total;

#if GENERATE_GIF
    checkCudaErrors(cudaMemcpy(
        x,
        d_x_arr,
        (size_t)n * sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        y,
        d_y_arr,
        (size_t)n * sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    ge_GIF *gif = NULL;

    gif = ge_new_gif(
        GIF_FILE,
        (uint16_t)FRAME_WIDTH,
        (uint16_t)FRAME_HEIGHT,
        palette,
        8,
        -1,
        0
    );

    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        render_frame_gif(gif, x, y, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++) {
        const int should_log = log_steps || (step + 1 == nsteps);

        update_velocities_and_positions_kernel<<<update_grid_size, update_block_size>>>(
            d_x_arr,
            d_y_arr,
            d_vx_arr,
            d_vy_arr,
            d_fx_arr,
            d_fy_arr,
            n,
            d_temp_x,
            d_temp_y,
            d_needs_rebuild
        );
        checkCudaErrors(cudaGetLastError());

        const double step_potential = compute_forces_verlet_cuda(
            d_x_arr,
            d_y_arr,
            d_vx_arr,
            d_vy_arr,
            d_fx_arr,
            d_fy_arr,
            n,
            d_particle_cell_indices,
            d_particle_ids,
            d_cell_start,
            d_next_pos,
            d_temp_x,
            d_temp_y,
            d_temp_vx,
            d_temp_vy,
            d_temp_particle_cell_indices,
            d_needs_rebuild,
            d_cell_neighbors,
            d_neighbor_counts,
            d_neighbor_list,
            num_cells,
            r_skin,
            d_pe_partial,
            force_blocks,
            force_block_size,
            should_log
        );

        update_velocities_second_half_kernel<<<update_grid_size, update_block_size>>>(
            d_vx_arr,
            d_vy_arr,
            d_fx_arr,
            d_fy_arr,
            n
        );
        checkCudaErrors(cudaGetLastError());

        if (should_log) {
            out.final_potential = step_potential;

            out.final_kinetic = compute_ke_cuda(
                d_vx_arr,
                d_vy_arr,
                n,
                d_ke_partial,
                num_ke_blocks,
                ke_block_size
            );

            out.final_total = out.final_kinetic + out.final_potential;
        }

        if (log_steps) {
            printf(
                "step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                step,
                out.final_kinetic,
                out.final_potential,
                out.final_total
            );
        }

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            checkCudaErrors(cudaMemcpy(
                x,
                d_x_arr,
                (size_t)n * sizeof(double),
                cudaMemcpyDeviceToHost
            ));

            checkCudaErrors(cudaMemcpy(
                y,
                d_y_arr,
                (size_t)n * sizeof(double),
                cudaMemcpyDeviceToHost
            ));

            render_frame_gif(gif, x, y, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

    checkCudaErrors(cudaMemcpy(x, d_x_arr, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y, d_y_arr, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(vx, d_vx_arr, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(vy, d_vy_arr, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(fx, d_fx_arr, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(fy, d_fy_arr, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    checkCudaErrors(cudaFree(d_x_arr));
    checkCudaErrors(cudaFree(d_y_arr));
    checkCudaErrors(cudaFree(d_vx_arr));
    checkCudaErrors(cudaFree(d_vy_arr));
    checkCudaErrors(cudaFree(d_fx_arr));
    checkCudaErrors(cudaFree(d_fy_arr));

    checkCudaErrors(cudaFree(d_temp_x));
    checkCudaErrors(cudaFree(d_temp_y));
    checkCudaErrors(cudaFree(d_temp_vx));
    checkCudaErrors(cudaFree(d_temp_vy));

    checkCudaErrors(cudaFree(d_particle_cell_indices));
    checkCudaErrors(cudaFree(d_particle_ids));
    checkCudaErrors(cudaFree(d_cell_start));
    checkCudaErrors(cudaFree(d_next_pos));
    checkCudaErrors(cudaFree(d_temp_particle_cell_indices));
    checkCudaErrors(cudaFree(d_needs_rebuild));

    checkCudaErrors(cudaFree(d_cell_neighbors));

    checkCudaErrors(cudaFree(d_neighbor_counts));
    checkCudaErrors(cudaFree(d_neighbor_list));

    checkCudaErrors(cudaFree(d_ke_partial));
    checkCudaErrors(cudaFree(d_pe_partial));

    return out;
}