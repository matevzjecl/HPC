#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <utility>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#include "lennard-jones.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#define NEIGHBOR_COUNT 27

typedef float real;
#define REAL_C(x) ((real)(x))

#ifndef R_SKIN
#define R_SKIN 0.2f
#endif

#ifndef FORCE_BLOCK_SIZE
#define FORCE_BLOCK_SIZE 256
#endif

#ifndef KE_BLOCK_SIZE
#define KE_BLOCK_SIZE 32
#endif

#ifndef UPDATE_BLOCK_SIZE
#define UPDATE_BLOCK_SIZE 256
#endif

#ifndef REBUILD_BLOCK_SIZE
#define REBUILD_BLOCK_SIZE 128
#endif

#ifndef LJ_MAX_NEIGHBORS
#define LJ_MAX_NEIGHBORS 1024
#endif

#ifndef LJ_LOG_MAX_NEIGHBORS
#define LJ_LOG_MAX_NEIGHBORS 0
#endif

#define R_CUT_REAL REAL_C(R_CUT)
#define SIGMA_REAL REAL_C(SIGMA)
#define EPSILON_REAL REAL_C(EPSILON)
#define DT_REAL REAL_C(DT)

#define R_CUT2 ((real)(R_CUT) * (real)(R_CUT))
#define SIGMA2 ((real)(SIGMA) * (real)(SIGMA))
#define FOUR_EPSILON ((real)4.0 * (real)(EPSILON))
#define TWENTYFOUR_EPSILON ((real)24.0 * (real)(EPSILON))
#define HALF_DT ((real)0.5 * (real)(DT))

#define LJ_SR (SIGMA_REAL / R_CUT_REAL)
#define LJ_SR2 (LJ_SR * LJ_SR)
#define LJ_SR6 (LJ_SR2 * LJ_SR2 * LJ_SR2)
#define LJ_SR12 (LJ_SR6 * LJ_SR6)
#define V_SHIFT (FOUR_EPSILON * (LJ_SR12 - LJ_SR6))

typedef struct {
    real box_size;
    real inv_box_size;
    real max_disp2;

    int num_cells_x;
    int num_cells_y;
    int num_cells_z;
    int num_cells;

    real cell_size_x;
    real cell_size_y;
    real cell_size_z;

    real inv_cell_size_x;
    real inv_cell_size_y;
    real inv_cell_size_z;

    int max_neighbors;
} constants3d;

__constant__ constants3d d_c;

static inline constants3d make_constants(double box_size, float r_skin, unsigned int n) {
    constants3d c;

    c.box_size = REAL_C(box_size);
    c.inv_box_size = REAL_C(1.0 / box_size);

    c.max_disp2 = REAL_C(0.25f) * REAL_C(r_skin) * REAL_C(r_skin);

    const double target_cell_size = (double)R_CUT + (double)r_skin;

    c.num_cells_x = (int)(box_size / target_cell_size);
    c.num_cells_y = (int)(box_size / target_cell_size);
    c.num_cells_z = (int)(box_size / target_cell_size);

    if (c.num_cells_x < 3) c.num_cells_x = 3;
    if (c.num_cells_y < 3) c.num_cells_y = 3;
    if (c.num_cells_z < 3) c.num_cells_z = 3;

    c.num_cells = c.num_cells_x * c.num_cells_y * c.num_cells_z;

    c.cell_size_x = REAL_C(box_size / (double)c.num_cells_x);
    c.cell_size_y = REAL_C(box_size / (double)c.num_cells_y);
    c.cell_size_z = REAL_C(box_size / (double)c.num_cells_z);

    c.inv_cell_size_x = REAL_C(1.0) / c.cell_size_x;
    c.inv_cell_size_y = REAL_C(1.0) / c.cell_size_y;
    c.inv_cell_size_z = REAL_C(1.0) / c.cell_size_z;

    /*
        Important fix:
        Old version used n - 1, which makes the neighbor list O(n^2).
        For n = 100000 this would allocate around 40 GB just for d_neighbor_list.

        This caps the stored neighbor list to a practical fixed size.
        The temporary debug array d_neighbor_total_counts still records the true count.
    */
    if (n <= 1) {
        c.max_neighbors = 1;
    } else {
        const int full_max = (int)n - 1;
        c.max_neighbors = full_max < LJ_MAX_NEIGHBORS ? full_max : LJ_MAX_NEIGHBORS;

        if (c.max_neighbors < 1) {
            c.max_neighbors = 1;
        }
    }

    return c;
}

double random_double(void) {
    static const double inv_rand_max = 1.0 / (double)RAND_MAX;
    return (double)rand() * inv_rand_max;
}

// static double relative_change(double current, double previous) {
//     const double eps = 1e-12;
//
//     if (fabs(previous) < eps) {
//         return 0.0;
//     }
//
//     return (current - previous) / previous;
// }

static int is_power_of_two_int(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

static int sanitize_block_size(int value, int fallback) {
    if (value <= 0) {
        return fallback;
    }

    if (value > 1024) {
        fprintf(stderr, "Warning: block size %d too large, using %d\n", value, fallback);
        return fallback;
    }

    if (!is_power_of_two_int(value)) {
        fprintf(stderr, "Warning: block size %d is not power of two, using %d\n", value, fallback);
        return fallback;
    }

    return value;
}

static int get_env_int_or_default(const char *name, int default_value) {
    const char *value = getenv(name);

    if (value == NULL || value[0] == '\0') {
        return default_value;
    }

    char *end = NULL;
    long parsed = strtol(value, &end, 10);

    if (end == value || *end != '\0') {
        fprintf(stderr, "Warning: invalid %s=%s, using %d\n", name, value, default_value);
        return default_value;
    }

    return sanitize_block_size((int)parsed, default_value);
}

static float get_env_float_or_default(const char *name, float default_value) {
    const char *value = getenv(name);

    if (value == NULL || value[0] == '\0') {
        return default_value;
    }

    char *end = NULL;
    float parsed = strtof(value, &end);

    if (end == value || *end != '\0' || parsed <= 0.0f) {
        fprintf(stderr, "Warning: invalid %s=%s, using %.4f\n", name, value, default_value);
        return default_value;
    }

    return parsed;
}

static void enable_mapped_host_memory(void) {
    cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);

    if (status != cudaSuccess && status != cudaErrorSetOnActiveProcess) {
        checkCudaErrors(status);
    }
}

int initialize_particles(
    Particle *particles,
    unsigned int n,
    double box_size,
    double placement_fraction,
    unsigned int seed,
    double temperature
) {
    srand(seed);

    unsigned int n_side = (unsigned int)ceil(cbrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    double mean_vz = 0.0;

    for (unsigned int k = 0; k < n; k++) {
        particles[k].id = k;

        unsigned int ix = k % n_side;
        unsigned int iy = (k / n_side) % n_side;
        unsigned int iz = k / (n_side * n_side);

        double x0 = offset + (0.5 + (double)ix) * delta;
        double y0 = offset + (0.5 + (double)iy) * delta;
        double z0 = offset + (0.5 + (double)iz) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].z = z0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        particles[k].vz = 2.0 * random_double() - 1.0;

        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
        mean_vz += particles[k].vz;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    mean_vz /= (double)n;

    double ke = 0.0;

    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        particles[k].vz -= mean_vz;

        ke += 0.5 * (
            particles[k].vx * particles[k].vx +
            particles[k].vy * particles[k].vy +
            particles[k].vz * particles[k].vz
        );
    }

    double current_temperature = ke / (double)n;

    if (current_temperature <= 0.0) {
        return 0;
    }

    double scale = sqrt(temperature / current_temperature);

    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
        particles[k].vz *= scale;
    }

    return 1;
}

__device__ __forceinline__ real minimum_image(real d) {
    const real half_box = REAL_C(0.5) * d_c.box_size;

    if (d > half_box) {
        d -= d_c.box_size;
    } else if (d < -half_box) {
        d += d_c.box_size;
    }

    return d;
}

__device__ __forceinline__ real wrap_position(real value) {
    if (value >= d_c.box_size) {
        value -= d_c.box_size;
    } else if (value < REAL_C(0.0)) {
        value += d_c.box_size;
    }

    return value;
}

__device__ __forceinline__ real fast_inv_r2(real r2) {
    return __fdividef(REAL_C(1.0), r2);
}

__global__ void compute_ke_partial_kernel(
    const real *__restrict__ vx,
    const real *__restrict__ vy,
    const real *__restrict__ vz,
    unsigned int n,
    real *__restrict__ partials
) {
    extern __shared__ real shared_ke[];

    const unsigned int tid = threadIdx.x;
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    real local_ke = REAL_C(0.0);

    for (unsigned int i = global_id; i < n; i += stride) {
        local_ke += REAL_C(0.5) * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
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
    const real *__restrict__ d_vx_arr,
    const real *__restrict__ d_vy_arr,
    const real *__restrict__ d_vz_arr,
    unsigned int n,
    real *__restrict__ d_ke_partial,
    int num_ke_blocks,
    int ke_block_size
) {
    compute_ke_partial_kernel<<<
        num_ke_blocks,
        ke_block_size,
        ke_block_size * sizeof(real)
    >>>(
        d_vx_arr,
        d_vy_arr,
        d_vz_arr,
        n,
        d_ke_partial
    );
    checkCudaErrors(cudaGetLastError());

    thrust::device_ptr<real> d_ke_ptr(d_ke_partial);

    return thrust::reduce(
        d_ke_ptr,
        d_ke_ptr + num_ke_blocks,
        REAL_C(0.0),
        thrust::plus<real>()
    );
}

__global__ void count_cells_kernel(
    const real *__restrict__ x,
    const real *__restrict__ y,
    const real *__restrict__ z,
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
    int cell_z = (int)(z[i] * d_c.inv_cell_size_z);

    if (cell_x >= d_c.num_cells_x) cell_x = d_c.num_cells_x - 1;
    if (cell_y >= d_c.num_cells_y) cell_y = d_c.num_cells_y - 1;
    if (cell_z >= d_c.num_cells_z) cell_z = d_c.num_cells_z - 1;

    if (cell_x < 0) cell_x = 0;
    if (cell_y < 0) cell_y = 0;
    if (cell_z < 0) cell_z = 0;

    const int cell =
        (cell_z * d_c.num_cells_y + cell_y) * d_c.num_cells_x + cell_x;

    particle_cell_indices[i] = cell;

    atomicAdd(&cell_start[cell + 1], 1);
}

__global__ void scatter_particles_kernel(
    const real *__restrict__ x,
    const real *__restrict__ y,
    const real *__restrict__ z,
    const real *__restrict__ vx,
    const real *__restrict__ vy,
    const real *__restrict__ vz,
    const int *__restrict__ particle_ids,
    unsigned int n,
    const int *__restrict__ particle_cell_indices,
    int *__restrict__ next_pos,
    real *__restrict__ temp_x,
    real *__restrict__ temp_y,
    real *__restrict__ temp_z,
    real *__restrict__ temp_vx,
    real *__restrict__ temp_vy,
    real *__restrict__ temp_vz,
    int *__restrict__ temp_particle_ids,
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
    temp_z[pos] = z[i];

    temp_vx[pos] = vx[i];
    temp_vy[pos] = vy[i];
    temp_vz[pos] = vz[i];

    temp_particle_ids[pos] = particle_ids[i];
    temp_particle_cell_indices[pos] = cell;
}

__global__ void build_verlet_list_from_sorted_particles_kernel(
    const real *__restrict__ x,
    const real *__restrict__ y,
    const real *__restrict__ z,
    const int *__restrict__ particle_cell_indices,
    const int *__restrict__ cell_start,
    const int *__restrict__ cell_neighbors,
    int *__restrict__ neighbor_counts,
    int *__restrict__ neighbor_total_counts,
    int *__restrict__ neighbor_list,
    unsigned int n,
    float r_skin,
    int *__restrict__ neighbor_overflow
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const real r_list = R_CUT_REAL + REAL_C(r_skin);
    const real r_list2 = r_list * r_list;

    const int max_neighbors = d_c.max_neighbors;

    const int cell_i = particle_cell_indices[i];

    const real xi = x[i];
    const real yi = y[i];
    const real zi = z[i];

    int stored_count = 0;
    int total_count = 0;

    for (int k = 0; k < NEIGHBOR_COUNT; ++k) {
        const int cell_j = cell_neighbors[cell_i * NEIGHBOR_COUNT + k];

        const int j_start = cell_start[cell_j];
        const int j_end = cell_start[cell_j + 1];

        for (int j = j_start; j < j_end; ++j) {
            if ((int)i == j) {
                continue;
            }

            real dx = xi - x[j];
            real dy = yi - y[j];
            real dz = zi - z[j];

            dx = minimum_image(dx);
            dy = minimum_image(dy);
            dz = minimum_image(dz);

            const real dist2 = dx * dx + dy * dy + dz * dz;

            if (dist2 < r_list2) {
                total_count++;

                if (stored_count < max_neighbors) {
                    neighbor_list[(size_t)stored_count * (size_t)n + (size_t)i] = j;
                    stored_count++;
                } else {
                    atomicExch(neighbor_overflow, 1);
                }
            }
        }
    }

    /*
        neighbor_counts is the stored/capped count used by the force kernel.
        neighbor_total_counts is the true count used only for temporary logging.
    */
    neighbor_counts[i] = stored_count;
    neighbor_total_counts[i] = total_count;
}

__global__ void compute_forces_verlet_kernel(
    const real *__restrict__ x,
    const real *__restrict__ y,
    const real *__restrict__ z,
    real *__restrict__ vx,
    real *__restrict__ vy,
    real *__restrict__ vz,
    real *__restrict__ fx,
    real *__restrict__ fy,
    real *__restrict__ fz,
    unsigned int n,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_list,
    real *__restrict__ pe_partial,
    int update_velocity_second_half
) {
    extern __shared__ real shared_pe[];

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    real pe_local = REAL_C(0.0);

    if (i < n) {
        const int count = neighbor_counts[i];

        real fxi = REAL_C(0.0);
        real fyi = REAL_C(0.0);
        real fzi = REAL_C(0.0);
        real pei = REAL_C(0.0);

        const real xi = x[i];
        const real yi = y[i];
        const real zi = z[i];

        for (int k = 0; k < count; ++k) {
            const int j = neighbor_list[(size_t)k * (size_t)n + (size_t)i];

            real dx = xi - x[j];
            real dy = yi - y[j];
            real dz = zi - z[j];

            dx = minimum_image(dx);
            dy = minimum_image(dy);
            dz = minimum_image(dz);

            const real r2 = dx * dx + dy * dy + dz * dz;

            if (r2 >= R_CUT2 || r2 == REAL_C(0.0)) {
                continue;
            }

            const real inv_r2 = fast_inv_r2(r2);
            const real sr2 = SIGMA2 * inv_r2;
            const real sr6 = sr2 * sr2 * sr2;
            const real sr12 = sr6 * sr6;

            const real fij =
                TWENTYFOUR_EPSILON * (REAL_C(2.0) * sr12 - sr6) * inv_r2;

            fxi += fij * dx;
            fyi += fij * dy;
            fzi += fij * dz;

            const real vij = FOUR_EPSILON * (sr12 - sr6) - V_SHIFT;

            pei += REAL_C(0.5) * vij;
        }

        fx[i] = fxi;
        fy[i] = fyi;
        fz[i] = fzi;

        if (update_velocity_second_half) {
            vx[i] += HALF_DT * fxi;
            vy[i] += HALF_DT * fyi;
            vz[i] += HALF_DT * fzi;
        }

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
    const real *__restrict__ x,
    const real *__restrict__ y,
    const real *__restrict__ z,
    real *__restrict__ vx,
    real *__restrict__ vy,
    real *__restrict__ vz,
    real *__restrict__ fx,
    real *__restrict__ fy,
    real *__restrict__ fz,
    unsigned int n,
    const int *__restrict__ neighbor_counts,
    const int *__restrict__ neighbor_list,
    int update_velocity_second_half
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const int count = neighbor_counts[i];

    real fxi = REAL_C(0.0);
    real fyi = REAL_C(0.0);
    real fzi = REAL_C(0.0);

    const real xi = x[i];
    const real yi = y[i];
    const real zi = z[i];

    for (int k = 0; k < count; ++k) {
        const int j = neighbor_list[(size_t)k * (size_t)n + (size_t)i];

        real dx = xi - x[j];
        real dy = yi - y[j];
        real dz = zi - z[j];

        dx = minimum_image(dx);
        dy = minimum_image(dy);
        dz = minimum_image(dz);

        const real r2 = dx * dx + dy * dy + dz * dz;

        if (r2 >= R_CUT2 || r2 == REAL_C(0.0)) {
            continue;
        }

        const real inv_r2 = fast_inv_r2(r2);
        const real sr2 = SIGMA2 * inv_r2;
        const real sr6 = sr2 * sr2 * sr2;
        const real sr12 = sr6 * sr6;

        const real fij =
            TWENTYFOUR_EPSILON * (REAL_C(2.0) * sr12 - sr6) * inv_r2;

        fxi += fij * dx;
        fyi += fij * dy;
        fzi += fij * dz;
    }

    fx[i] = fxi;
    fy[i] = fyi;
    fz[i] = fzi;

    if (update_velocity_second_half) {
        vx[i] += HALF_DT * fxi;
        vy[i] += HALF_DT * fyi;
        vz[i] += HALF_DT * fzi;
    }
}

__global__ void update_velocities_and_positions_kernel(
    real *__restrict__ x,
    real *__restrict__ y,
    real *__restrict__ z,
    real *__restrict__ vx,
    real *__restrict__ vy,
    real *__restrict__ vz,
    const real *__restrict__ fx,
    const real *__restrict__ fy,
    const real *__restrict__ fz,
    unsigned int n,
    const real *__restrict__ temp_x,
    const real *__restrict__ temp_y,
    const real *__restrict__ temp_z,
    int *__restrict__ needs_rebuild
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const real max_disp2 = d_c.max_disp2;

    vx[i] += HALF_DT * fx[i];
    vy[i] += HALF_DT * fy[i];
    vz[i] += HALF_DT * fz[i];

    x[i] += DT_REAL * vx[i];
    y[i] += DT_REAL * vy[i];
    z[i] += DT_REAL * vz[i];

    x[i] = wrap_position(x[i]);
    y[i] = wrap_position(y[i]);
    z[i] = wrap_position(z[i]);

    real dx = x[i] - temp_x[i];
    real dy = y[i] - temp_y[i];
    real dz = z[i] - temp_z[i];

    dx = minimum_image(dx);
    dy = minimum_image(dy);
    dz = minimum_image(dz);

    const real dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 > max_disp2) {
        atomicExch(needs_rebuild, 1);
    }
}

void build_cell_neighbors_full(
    int *__restrict__ cell_neighbors,
    const constants3d *__restrict__ c
) {
    const int num_cells = c->num_cells;
    const int num_cells_x = c->num_cells_x;
    const int num_cells_y = c->num_cells_y;
    const int num_cells_z = c->num_cells_z;

    for (int cell = 0; cell < num_cells; ++cell) {
        const int cx = cell % num_cells_x;
        const int cy = (cell / num_cells_x) % num_cells_y;
        const int cz = cell / (num_cells_x * num_cells_y);

        int k = 0;

        for (int oz = -1; oz <= 1; ++oz) {
            for (int oy = -1; oy <= 1; ++oy) {
                for (int ox = -1; ox <= 1; ++ox) {
                    int nx = cx + ox;
                    int ny = cy + oy;
                    int nz = cz + oz;

                    if (nx < 0) nx += num_cells_x;
                    else if (nx >= num_cells_x) nx -= num_cells_x;

                    if (ny < 0) ny += num_cells_y;
                    else if (ny >= num_cells_y) ny -= num_cells_y;

                    if (nz < 0) nz += num_cells_z;
                    else if (nz >= num_cells_z) nz -= num_cells_z;

                    cell_neighbors[cell * NEIGHBOR_COUNT + k] =
                        (nz * num_cells_y + ny) * num_cells_x + nx;

                    k++;
                }
            }
        }
    }
}

void rebuild_cells_cuda(
    real *&d_x_arr,
    real *&d_y_arr,
    real *&d_z_arr,
    real *&d_vx_arr,
    real *&d_vy_arr,
    real *&d_vz_arr,
    unsigned int n,
    int *&d_particle_cell_indices,
    int *&d_particle_ids,
    int *__restrict__ d_cell_start,
    int *__restrict__ d_next_pos,
    real *&d_temp_x,
    real *&d_temp_y,
    real *&d_temp_z,
    real *&d_temp_vx,
    real *&d_temp_vy,
    real *&d_temp_vz,
    int *&d_temp_particle_ids,
    int *&d_temp_particle_cell_indices,
    int num_cells,
    int rebuild_block_size
) {
    const int block_size = sanitize_block_size(rebuild_block_size, REBUILD_BLOCK_SIZE);
    const int grid_size = (int)((n + block_size - 1) / block_size);

    checkCudaErrors(cudaMemset(
        d_cell_start,
        0,
        (size_t)(num_cells + 1) * sizeof(int)
    ));

    count_cells_kernel<<<grid_size, block_size>>>(
        d_x_arr,
        d_y_arr,
        d_z_arr,
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
        d_z_arr,
        d_vx_arr,
        d_vy_arr,
        d_vz_arr,
        d_particle_ids,
        n,
        d_particle_cell_indices,
        d_next_pos,
        d_temp_x,
        d_temp_y,
        d_temp_z,
        d_temp_vx,
        d_temp_vy,
        d_temp_vz,
        d_temp_particle_ids,
        d_temp_particle_cell_indices
    );
    checkCudaErrors(cudaGetLastError());

    std::swap(d_x_arr, d_temp_x);
    std::swap(d_y_arr, d_temp_y);
    std::swap(d_z_arr, d_temp_z);
    std::swap(d_vx_arr, d_temp_vx);
    std::swap(d_vy_arr, d_temp_vy);
    std::swap(d_vz_arr, d_temp_vz);
    std::swap(d_particle_ids, d_temp_particle_ids);
    std::swap(d_particle_cell_indices, d_temp_particle_cell_indices);

    checkCudaErrors(cudaMemcpy(
        d_temp_x,
        d_x_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_temp_y,
        d_y_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_temp_z,
        d_z_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToDevice
    ));
}

static void build_verlet_list_cuda(
    const real *__restrict__ d_x_arr,
    const real *__restrict__ d_y_arr,
    const real *__restrict__ d_z_arr,
    const int *__restrict__ d_particle_cell_indices,
    const int *__restrict__ d_cell_start,
    const int *__restrict__ d_cell_neighbors,
    int *__restrict__ d_neighbor_counts,
    int *__restrict__ d_neighbor_total_counts,
    int *__restrict__ d_neighbor_list,
    int *__restrict__ d_neighbor_overflow,
    unsigned int n,
    float r_skin,
    int rebuild_block_size
) {
    const int block_size = sanitize_block_size(rebuild_block_size, REBUILD_BLOCK_SIZE);
    const int grid_size = (int)((n + block_size - 1) / block_size);

    checkCudaErrors(cudaMemset(d_neighbor_overflow, 0, sizeof(int)));

    build_verlet_list_from_sorted_particles_kernel<<<grid_size, block_size>>>(
        d_x_arr,
        d_y_arr,
        d_z_arr,
        d_particle_cell_indices,
        d_cell_start,
        d_cell_neighbors,
        d_neighbor_counts,
        d_neighbor_total_counts,
        d_neighbor_list,
        n,
        r_skin,
        d_neighbor_overflow
    );
    checkCudaErrors(cudaGetLastError());

#if LJ_LOG_MAX_NEIGHBORS
    thrust::device_ptr<int> total_counts_ptr(d_neighbor_total_counts);

    const int h_max_neighbors_seen = thrust::reduce(
        total_counts_ptr,
        total_counts_ptr + n,
        0,
        thrust::maximum<int>()
    );

    int h_neighbor_overflow = 0;

    checkCudaErrors(cudaMemcpy(
        &h_neighbor_overflow,
        d_neighbor_overflow,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    const int stored_cap =
        n <= 1
            ? 1
            : (((int)n - 1) < LJ_MAX_NEIGHBORS ? ((int)n - 1) : LJ_MAX_NEIGHBORS);

    fprintf(
        stderr,
        "Neighbor debug: true_max_neighbors=%d stored_cap=%d LJ_MAX_NEIGHBORS=%d overflow=%s\n",
        h_max_neighbors_seen,
        stored_cap,
        LJ_MAX_NEIGHBORS,
        h_neighbor_overflow ? "YES" : "no"
    );

    if (h_neighbor_overflow) {
        fprintf(
            stderr,
            "Warning: neighbor list overflow. Increase LJ_MAX_NEIGHBORS. Current LJ_MAX_NEIGHBORS=%d true_max_neighbors=%d\n",
            LJ_MAX_NEIGHBORS,
            h_max_neighbors_seen
        );
    }
#else
    int h_neighbor_overflow = 0;

    checkCudaErrors(cudaMemcpy(
        &h_neighbor_overflow,
        d_neighbor_overflow,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (h_neighbor_overflow) {
        fprintf(
            stderr,
            "Warning: neighbor list overflow. Increase LJ_MAX_NEIGHBORS. Current LJ_MAX_NEIGHBORS=%d\n",
            LJ_MAX_NEIGHBORS
        );
    }
#endif
}

double compute_forces_verlet_cuda(
    real *&d_x_arr,
    real *&d_y_arr,
    real *&d_z_arr,
    real *&d_vx_arr,
    real *&d_vy_arr,
    real *&d_vz_arr,
    real *__restrict__ d_fx_arr,
    real *__restrict__ d_fy_arr,
    real *__restrict__ d_fz_arr,
    unsigned int n,
    int *&d_particle_cell_indices,
    int *&d_particle_ids,
    int *__restrict__ d_cell_start,
    int *__restrict__ d_next_pos,
    real *&d_temp_x,
    real *&d_temp_y,
    real *&d_temp_z,
    real *&d_temp_vx,
    real *&d_temp_vy,
    real *&d_temp_vz,
    int *&d_temp_particle_ids,
    int *&d_temp_particle_cell_indices,
    int *__restrict__ d_needs_rebuild,
    int *__restrict__ h_needs_rebuild,
    const int *__restrict__ d_cell_neighbors,
    int *__restrict__ d_neighbor_counts,
    int *__restrict__ d_neighbor_total_counts,
    int *__restrict__ d_neighbor_list,
    int *__restrict__ d_neighbor_overflow,
    int num_cells,
    float r_skin,
    real *__restrict__ d_pe_partial,
    int force_blocks,
    int force_block_size,
    int compute_energy,
    int rebuild_block_size,
    int update_velocity_second_half
) {
    (void)d_needs_rebuild;

    checkCudaErrors(cudaDeviceSynchronize());

    if (*h_needs_rebuild) {
        rebuild_cells_cuda(
            d_x_arr,
            d_y_arr,
            d_z_arr,
            d_vx_arr,
            d_vy_arr,
            d_vz_arr,
            n,
            d_particle_cell_indices,
            d_particle_ids,
            d_cell_start,
            d_next_pos,
            d_temp_x,
            d_temp_y,
            d_temp_z,
            d_temp_vx,
            d_temp_vy,
            d_temp_vz,
            d_temp_particle_ids,
            d_temp_particle_cell_indices,
            num_cells,
            rebuild_block_size
        );

        build_verlet_list_cuda(
            d_x_arr,
            d_y_arr,
            d_z_arr,
            d_particle_cell_indices,
            d_cell_start,
            d_cell_neighbors,
            d_neighbor_counts,
            d_neighbor_total_counts,
            d_neighbor_list,
            d_neighbor_overflow,
            n,
            r_skin,
            rebuild_block_size
        );

        *h_needs_rebuild = 0;
    }

    if (compute_energy) {
        compute_forces_verlet_kernel<<<
            force_blocks,
            force_block_size,
            force_block_size * sizeof(real)
        >>>(
            d_x_arr,
            d_y_arr,
            d_z_arr,
            d_vx_arr,
            d_vy_arr,
            d_vz_arr,
            d_fx_arr,
            d_fy_arr,
            d_fz_arr,
            n,
            d_neighbor_counts,
            d_neighbor_list,
            d_pe_partial,
            update_velocity_second_half
        );
        checkCudaErrors(cudaGetLastError());

        thrust::device_ptr<real> d_pe_ptr(d_pe_partial);

        return thrust::reduce(
            d_pe_ptr,
            d_pe_ptr + force_blocks,
            REAL_C(0.0),
            thrust::plus<real>()
        );
    }

    compute_forces_verlet_no_energy_kernel<<<
        force_blocks,
        force_block_size
    >>>(
        d_x_arr,
        d_y_arr,
        d_z_arr,
        d_vx_arr,
        d_vy_arr,
        d_vz_arr,
        d_fx_arr,
        d_fy_arr,
        d_fz_arr,
        n,
        d_neighbor_counts,
        d_neighbor_list,
        update_velocity_second_half
    );
    checkCudaErrors(cudaGetLastError());

    return 0.0;
}

static void *checked_malloc(size_t bytes, const char *name) {
    void *ptr = malloc(bytes);

    if (ptr == NULL) {
        fprintf(stderr, "Allocation failed for %s\n", name);
        exit(EXIT_FAILURE);
    }

    return ptr;
}

static SimulationResult run_simulation_cuda_internal(
    Particle *particles,
    unsigned int n,
    unsigned int nsteps,
    double box_size,
    int log_steps,
    float r_skin,
    int force_block_size,
    int ke_block_size,
    int update_block_size,
    int rebuild_block_size
) {
    SimulationResult out;

    out.n = n;
    out.particles = particles;
    out.start_kinetic = 0.0;
    out.start_potential = 0.0;
    out.start_total = 0.0;
    out.final_kinetic = 0.0;
    out.final_potential = 0.0;
    out.final_total = 0.0;

    if (n == 0) {
        return out;
    }

    enable_mapped_host_memory();

    force_block_size = sanitize_block_size(force_block_size, FORCE_BLOCK_SIZE);
    ke_block_size = sanitize_block_size(ke_block_size, KE_BLOCK_SIZE);
    update_block_size = sanitize_block_size(update_block_size, UPDATE_BLOCK_SIZE);
    rebuild_block_size = sanitize_block_size(rebuild_block_size, REBUILD_BLOCK_SIZE);

    real *h_x = (real *)checked_malloc((size_t)n * sizeof(real), "h_x");
    real *h_y = (real *)checked_malloc((size_t)n * sizeof(real), "h_y");
    real *h_z = (real *)checked_malloc((size_t)n * sizeof(real), "h_z");
    real *h_vx = (real *)checked_malloc((size_t)n * sizeof(real), "h_vx");
    real *h_vy = (real *)checked_malloc((size_t)n * sizeof(real), "h_vy");
    real *h_vz = (real *)checked_malloc((size_t)n * sizeof(real), "h_vz");
    real *h_fx = (real *)checked_malloc((size_t)n * sizeof(real), "h_fx");
    real *h_fy = (real *)checked_malloc((size_t)n * sizeof(real), "h_fy");
    real *h_fz = (real *)checked_malloc((size_t)n * sizeof(real), "h_fz");
    int *h_particle_ids = (int *)checked_malloc((size_t)n * sizeof(int), "h_particle_ids");

    for (unsigned int i = 0; i < n; ++i) {
        h_x[i] = REAL_C(particles[i].x);
        h_y[i] = REAL_C(particles[i].y);
        h_z[i] = REAL_C(particles[i].z);

        h_vx[i] = REAL_C(particles[i].vx);
        h_vy[i] = REAL_C(particles[i].vy);
        h_vz[i] = REAL_C(particles[i].vz);

        h_particle_ids[i] = (int)i;
    }

    const constants3d h_c = make_constants(box_size, r_skin, n);
    const int num_cells = h_c.num_cells;
    const int max_neighbors = h_c.max_neighbors;

    checkCudaErrors(cudaMemcpyToSymbol(
        d_c,
        &h_c,
        sizeof(constants3d)
    ));

    const int update_grid_size = (int)((n + update_block_size - 1) / update_block_size);
    const int force_blocks = (int)((n + force_block_size - 1) / force_block_size);

    int num_ke_blocks = (int)((n + ke_block_size - 1) / ke_block_size);

    if (num_ke_blocks < 1) {
        num_ke_blocks = 1;
    }

    if (num_ke_blocks > 1024) {
        num_ke_blocks = 1024;
    }

    real *d_x_arr = NULL;
    real *d_y_arr = NULL;
    real *d_z_arr = NULL;
    real *d_vx_arr = NULL;
    real *d_vy_arr = NULL;
    real *d_vz_arr = NULL;
    real *d_fx_arr = NULL;
    real *d_fy_arr = NULL;
    real *d_fz_arr = NULL;

    real *d_temp_x = NULL;
    real *d_temp_y = NULL;
    real *d_temp_z = NULL;
    real *d_temp_vx = NULL;
    real *d_temp_vy = NULL;
    real *d_temp_vz = NULL;

    real *d_ke_partial = NULL;
    real *d_pe_partial = NULL;

    int *d_particle_cell_indices = NULL;
    int *d_particle_ids = NULL;
    int *d_temp_particle_ids = NULL;
    int *d_cell_start = NULL;
    int *d_next_pos = NULL;
    int *d_temp_particle_cell_indices = NULL;
    int *h_needs_rebuild = NULL;
    int *d_needs_rebuild = NULL;
    int *d_cell_neighbors = NULL;

    int *d_neighbor_counts = NULL;
    int *d_neighbor_total_counts = NULL;
    int *d_neighbor_list = NULL;
    int *d_neighbor_overflow = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_x_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_y_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_z_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_vx_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_vy_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_vz_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_fx_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_fy_arr, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_fz_arr, (size_t)n * sizeof(real)));

    checkCudaErrors(cudaMalloc((void **)&d_temp_x, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_y, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_z, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_vx, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_vy, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_vz, (size_t)n * sizeof(real)));

    checkCudaErrors(cudaMalloc((void **)&d_particle_cell_indices, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_particle_ids, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_particle_ids, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_cell_start, (size_t)(num_cells + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_next_pos, (size_t)num_cells * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_temp_particle_cell_indices, (size_t)n * sizeof(int)));

    checkCudaErrors(cudaHostAlloc(
        (void **)&h_needs_rebuild,
        sizeof(int),
        cudaHostAllocMapped
    ));

    checkCudaErrors(cudaHostGetDevicePointer(
        (void **)&d_needs_rebuild,
        h_needs_rebuild,
        0
    ));

    checkCudaErrors(cudaMalloc((void **)&d_neighbor_counts, (size_t)n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_neighbor_total_counts, (size_t)n * sizeof(int)));

    checkCudaErrors(cudaMalloc(
        (void **)&d_neighbor_list,
        (size_t)n * (size_t)max_neighbors * sizeof(int)
    ));

    checkCudaErrors(cudaMalloc((void **)&d_neighbor_overflow, sizeof(int)));

    checkCudaErrors(cudaMalloc((void **)&d_ke_partial, (size_t)num_ke_blocks * sizeof(real)));
    checkCudaErrors(cudaMalloc((void **)&d_pe_partial, (size_t)force_blocks * sizeof(real)));

    checkCudaErrors(cudaMemcpy(d_x_arr, h_x, (size_t)n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y_arr, h_y, (size_t)n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_z_arr, h_z, (size_t)n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vx_arr, h_vx, (size_t)n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vy_arr, h_vy, (size_t)n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vz_arr, h_vz, (size_t)n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_particle_ids, h_particle_ids, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_fx_arr, 0, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMemset(d_fy_arr, 0, (size_t)n * sizeof(real)));
    checkCudaErrors(cudaMemset(d_fz_arr, 0, (size_t)n * sizeof(real)));

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

    *h_needs_rebuild = 1;

    out.start_potential = compute_forces_verlet_cuda(
        d_x_arr,
        d_y_arr,
        d_z_arr,
        d_vx_arr,
        d_vy_arr,
        d_vz_arr,
        d_fx_arr,
        d_fy_arr,
        d_fz_arr,
        n,
        d_particle_cell_indices,
        d_particle_ids,
        d_cell_start,
        d_next_pos,
        d_temp_x,
        d_temp_y,
        d_temp_z,
        d_temp_vx,
        d_temp_vy,
        d_temp_vz,
        d_temp_particle_ids,
        d_temp_particle_cell_indices,
        d_needs_rebuild,
        h_needs_rebuild,
        d_cell_neighbors,
        d_neighbor_counts,
        d_neighbor_total_counts,
        d_neighbor_list,
        d_neighbor_overflow,
        num_cells,
        r_skin,
        d_pe_partial,
        force_blocks,
        force_block_size,
        1,
        rebuild_block_size,
        0
    );

    out.start_kinetic = compute_ke_cuda(
        d_vx_arr,
        d_vy_arr,
        d_vz_arr,
        n,
        d_ke_partial,
        num_ke_blocks,
        ke_block_size
    );

    out.start_total = out.start_kinetic + out.start_potential;

    out.final_potential = out.start_potential;
    out.final_kinetic = out.start_kinetic;
    out.final_total = out.start_total;

    for (unsigned int step = 0; step < nsteps; step++) {
        const int should_log = log_steps || (step + 1 == nsteps);

        update_velocities_and_positions_kernel<<<update_grid_size, update_block_size>>>(
            d_x_arr,
            d_y_arr,
            d_z_arr,
            d_vx_arr,
            d_vy_arr,
            d_vz_arr,
            d_fx_arr,
            d_fy_arr,
            d_fz_arr,
            n,
            d_temp_x,
            d_temp_y,
            d_temp_z,
            d_needs_rebuild
        );
        checkCudaErrors(cudaGetLastError());

        const double step_potential = compute_forces_verlet_cuda(
            d_x_arr,
            d_y_arr,
            d_z_arr,
            d_vx_arr,
            d_vy_arr,
            d_vz_arr,
            d_fx_arr,
            d_fy_arr,
            d_fz_arr,
            n,
            d_particle_cell_indices,
            d_particle_ids,
            d_cell_start,
            d_next_pos,
            d_temp_x,
            d_temp_y,
            d_temp_z,
            d_temp_vx,
            d_temp_vy,
            d_temp_vz,
            d_temp_particle_ids,
            d_temp_particle_cell_indices,
            d_needs_rebuild,
            h_needs_rebuild,
            d_cell_neighbors,
            d_neighbor_counts,
            d_neighbor_total_counts,
            d_neighbor_list,
            d_neighbor_overflow,
            num_cells,
            r_skin,
            d_pe_partial,
            force_blocks,
            force_block_size,
            should_log,
            rebuild_block_size,
            1
        );

        if (should_log) {
            out.final_potential = step_potential;

            out.final_kinetic = compute_ke_cuda(
                d_vx_arr,
                d_vy_arr,
                d_vz_arr,
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
    }

    checkCudaErrors(cudaMemcpy(
        h_x,
        d_x_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_y,
        d_y_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_z,
        d_z_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_vx,
        d_vx_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_vy,
        d_vy_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_vz,
        d_vz_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_fx,
        d_fx_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_fy,
        d_fy_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_fz,
        d_fz_arr,
        (size_t)n * sizeof(real),
        cudaMemcpyDeviceToHost
    ));

    checkCudaErrors(cudaMemcpy(
        h_particle_ids,
        d_particle_ids,
        (size_t)n * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    for (unsigned int i = 0; i < n; ++i) {
        int id = h_particle_ids[i];

        if (id < 0 || id >= (int)n) {
            id = (int)i;
        }

        particles[id].id = (unsigned int)id;

        particles[id].x = (double)h_x[i];
        particles[id].y = (double)h_y[i];
        particles[id].z = (double)h_z[i];

        particles[id].vx = (double)h_vx[i];
        particles[id].vy = (double)h_vy[i];
        particles[id].vz = (double)h_vz[i];

        particles[id].fx = (double)h_fx[i];
        particles[id].fy = (double)h_fy[i];
        particles[id].fz = (double)h_fz[i];
    }

    checkCudaErrors(cudaFree(d_x_arr));
    checkCudaErrors(cudaFree(d_y_arr));
    checkCudaErrors(cudaFree(d_z_arr));
    checkCudaErrors(cudaFree(d_vx_arr));
    checkCudaErrors(cudaFree(d_vy_arr));
    checkCudaErrors(cudaFree(d_vz_arr));
    checkCudaErrors(cudaFree(d_fx_arr));
    checkCudaErrors(cudaFree(d_fy_arr));
    checkCudaErrors(cudaFree(d_fz_arr));

    checkCudaErrors(cudaFree(d_temp_x));
    checkCudaErrors(cudaFree(d_temp_y));
    checkCudaErrors(cudaFree(d_temp_z));
    checkCudaErrors(cudaFree(d_temp_vx));
    checkCudaErrors(cudaFree(d_temp_vy));
    checkCudaErrors(cudaFree(d_temp_vz));

    checkCudaErrors(cudaFree(d_particle_cell_indices));
    checkCudaErrors(cudaFree(d_particle_ids));
    checkCudaErrors(cudaFree(d_temp_particle_ids));
    checkCudaErrors(cudaFree(d_cell_start));
    checkCudaErrors(cudaFree(d_next_pos));
    checkCudaErrors(cudaFree(d_temp_particle_cell_indices));
    checkCudaErrors(cudaFreeHost(h_needs_rebuild));

    checkCudaErrors(cudaFree(d_cell_neighbors));

    checkCudaErrors(cudaFree(d_neighbor_counts));
    checkCudaErrors(cudaFree(d_neighbor_total_counts));
    checkCudaErrors(cudaFree(d_neighbor_list));
    checkCudaErrors(cudaFree(d_neighbor_overflow));

    checkCudaErrors(cudaFree(d_ke_partial));
    checkCudaErrors(cudaFree(d_pe_partial));

    free(h_x);
    free(h_y);
    free(h_z);
    free(h_vx);
    free(h_vy);
    free(h_vz);
    free(h_fx);
    free(h_fy);
    free(h_fz);
    free(h_particle_ids);

    out.n = n;
    out.particles = particles;

    // printf("Change %f\n", relative_change(out.final_total, out.start_total));

    return out;
}

SimulationResult run_simulation_with_params(
    Particle *particles,
    unsigned int n,
    unsigned int nsteps,
    double box_size,
    int log_steps,
    float r_skin,
    int block_size
) {
    return run_simulation_cuda_internal(
        particles,
        n,
        nsteps,
        box_size,
        log_steps,
        r_skin,
        block_size,
        KE_BLOCK_SIZE,
        UPDATE_BLOCK_SIZE,
        REBUILD_BLOCK_SIZE
    );
}

SimulationResult run_simulation(
    Particle *particles,
    unsigned int n,
    unsigned int nsteps,
    double box_size,
    int log_steps
) {
    const float r_skin = get_env_float_or_default("LJ_R_SKIN", R_SKIN);

    const int force_block_size =
        get_env_int_or_default("LJ_FORCE_BLOCK_SIZE", FORCE_BLOCK_SIZE);

    const int ke_block_size =
        get_env_int_or_default("LJ_KE_BLOCK_SIZE", KE_BLOCK_SIZE);

    const int update_block_size =
        get_env_int_or_default("LJ_UPDATE_BLOCK_SIZE", UPDATE_BLOCK_SIZE);

    const int rebuild_block_size =
        get_env_int_or_default("LJ_REBUILD_BLOCK_SIZE", REBUILD_BLOCK_SIZE);

    if (getenv("LJ_PRINT_PARAMS") != NULL) {
        fprintf(
            stderr,
            "3D CUDA params: r_skin=%.4f force_block=%d ke_block=%d update_block=%d rebuild=%d max_neighbors=%d log_max_neighbors=%d\n",
            r_skin,
            force_block_size,
            ke_block_size,
            update_block_size,
            rebuild_block_size,
            LJ_MAX_NEIGHBORS,
            LJ_LOG_MAX_NEIGHBORS
        );
    }

    return run_simulation_cuda_internal(
        particles,
        n,
        nsteps,
        box_size,
        log_steps,
        r_skin,
        force_block_size,
        ke_block_size,
        update_block_size,
        rebuild_block_size
    );
}