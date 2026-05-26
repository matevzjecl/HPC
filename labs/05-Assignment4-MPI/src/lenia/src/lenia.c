#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include "mpi.h"
#include <string.h>

#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// #define GENERATE_GIF

#define HALO_TAG_UP 100
#define HALO_TAG_DOWN 101

typedef struct {
    int di;
    int dj;
} Offset;

typedef struct {
    unsigned int i;
    unsigned int j;
} Cell;

// Function to calculate Gaussian
static inline double gauss(double x, double mu, double sigma)
{
    double z = (x - mu) / sigma;
    return exp(-0.5 * z * z);
}

// Function for growth criteria
static inline double growth_lenia(double u)
{
    const double mu = 0.15;
    const double inv_sigma = 1.0 / 0.015;

    double z = (u - mu) * inv_sigma;
    return -1.0 + 2.0 * exp(-0.5 * z * z);
}

// Function to generate convolution kernel
double *generate_kernel(double *restrict K, const unsigned int size)
{
    const double mu = 0.5;
    const double sigma = 0.15;
    const int r = (int)size / 2;
    const int r2 = r * r;
    const double inv_r = 1.0 / (double)r;

    double sum = 0.0;

    if (K == NULL)
    {
        return K;
    }

    for (unsigned int y = 0; y < size; y++)
    {
        for (unsigned int x = 0; x < size; x++)
        {
            const int dy = (int)y - r;
            const int dx = (int)x - r;
            const int d2 = dx * dx + dy * dy;
            const unsigned int idx = y * size + x;

            if (d2 > r2)
            {
                K[idx] = 0.0;
                continue;
            }

            const double distance = sqrt((double)d2) * inv_r;
            K[idx] = gauss(distance, mu, sigma);
            sum += K[idx];
        }
    }

    if (sum != 0.0)
    {
        const double inv_sum = 1.0 / sum;

        for (unsigned int i = 0; i < size * size; i++)
        {
            K[i] *= inv_sum;
        }
    }

    return K;
}

static inline unsigned int wrap_unsigned_index(int index, unsigned int size)
{
    index %= (int)size;

    if (index < 0)
    {
        index += (int)size;
    }

    return (unsigned int)index;
}

// Full convolution, useful for serial/reference version
static inline double *convolve2d(
    double *restrict result,
    const double *restrict input,
    const double *restrict w,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int w_rows,
    const unsigned int w_cols
)
{
    if (result != NULL && input != NULL && w != NULL)
    {
        int row_center = (int)w_rows / 2;
        int col_center = (int)w_cols / 2;

        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                double sum = 0.0;

                for (unsigned int kr = 0; kr < w_rows; kr++)
                {
                    unsigned int input_row = wrap_unsigned_index(
                        (int)i - row_center + (int)kr,
                        rows
                    );

                    for (unsigned int kc = 0; kc < w_cols; kc++)
                    {
                        unsigned int input_col = wrap_unsigned_index(
                            (int)j - col_center + (int)kc,
                            cols
                        );

                        sum += w[kr * w_cols + kc] *
                               input[input_row * cols + input_col];
                    }
                }

                result[i * cols + j] = sum;
            }
        }
    }

    return result;
}

// Old full-world row convolution, kept for reference
static double *convolve2d_rows_local_result(
    double *restrict local_result,
    const double *restrict input,
    const double *restrict w,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int w_rows,
    const unsigned int w_cols,
    const unsigned int start_row,
    const unsigned int local_rows,
    const unsigned char *restrict active_local
)
{
    if (local_result == NULL || input == NULL || w == NULL || active_local == NULL)
    {
        return local_result;
    }

    const int row_center = (int)w_rows / 2;
    const int col_center = (int)w_cols / 2;

    for (unsigned int local_row = 0; local_row < local_rows; local_row++)
    {
        unsigned int global_row = start_row + local_row;

        for (unsigned int col = 0; col < cols; col++)
        {
            if (!active_local[local_row * cols + col])
            {
                local_result[local_row * cols + col] = 0.0;
                continue;
            }

            double sum = 0.0;

            for (unsigned int kr = 0; kr < w_rows; kr++)
            {
                unsigned int input_row = wrap_unsigned_index(
                    (int)global_row - row_center + (int)kr,
                    rows
                );

                for (unsigned int kc = 0; kc < w_cols; kc++)
                {
                    unsigned int input_col = wrap_unsigned_index(
                        (int)col - col_center + (int)kc,
                        cols
                    );

                    sum += w[kr * w_cols + kc] *
                           input[input_row * cols + input_col];
                }
            }

            local_result[local_row * cols + col] = sum;
        }
    }

    return local_result;
}

int build_circle_offsets(Offset *offsets, unsigned int kernel_size)
{
    int r = (int)kernel_size / 2;
    int count = 0;

    for (int di = -r; di < r; di++)
    {
        for (int dj = -r; dj < r; dj++)
        {
            if (di * di + dj * dj <= r * r)
            {
                offsets[count].di = di;
                offsets[count].dj = dj;
                count++;
            }
        }
    }

    return count;
}

void generate_mask_gather(const double *world,
                          unsigned char *active,
                          unsigned int rows,
                          unsigned int cols,
                          unsigned int kernel_size)
{
    int r = (int)kernel_size / 2;

    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            unsigned char is_active = 0;

            for (int di = -r; di < r && !is_active; di++)
            {
                for (int dj = -r; dj < r; dj++)
                {
                    if (di * di + dj * dj <= r * r)
                    {
                        int ii = ((int)i + di + (int)rows) % (int)rows;
                        int jj = ((int)j + dj + (int)cols) % (int)cols;

                        if (world[ii * cols + jj] > 0.0)
                        {
                            is_active = 1;
                            break;
                        }
                    }
                }
            }

            active[i * cols + j] = is_active;
        }
    }
}

void generate_mask_scatter(const double *world,
                           unsigned char *active,
                           unsigned int rows,
                           unsigned int cols,
                           unsigned int kernel_size)
{
    int r = (int)kernel_size / 2;

    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            if (world[i * cols + j] > 0.0)
            {
                for (int di = -r; di < r; di++)
                {
                    for (int dj = -r; dj < r; dj++)
                    {
                        if (di * di + dj * dj <= r * r)
                        {
                            int ii = ((int)i + di + (int)rows) % (int)rows;
                            int jj = ((int)j + dj + (int)cols) % (int)cols;

                            active[ii * cols + jj] = 1;
                        }
                    }
                }
            }
        }
    }
}

void generate_mask_scatter_fast(const double *world,
                                unsigned char *active,
                                unsigned int rows,
                                unsigned int cols,
                                const Offset *offsets,
                                int offset_count)
{
    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            if (world[i * cols + j] > 0.0)
            {
                for (int k = 0; k < offset_count; k++)
                {
                    int ii = (int)i + offsets[k].di;
                    int jj = (int)j + offsets[k].dj;

                    if (ii < 0) ii += rows;
                    else if (ii >= (int)rows) ii -= rows;

                    if (jj < 0) jj += cols;
                    else if (jj >= (int)cols) jj -= cols;

                    active[ii * cols + jj] = 1;
                }
            }
        }
    }
}

int collect_nonzero_cells(const double *world,
                          Cell *cells,
                          unsigned int rows,
                          unsigned int cols)
{
    int count = 0;

    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            if (world[i * cols + j] > 0.0)
            {
                cells[count].i = i;
                cells[count].j = j;
                count++;
            }
        }
    }

    return count;
}

int collect_nonzero_cells_local_rows(const double *world,
                                     Cell *cells,
                                     unsigned int start_row,
                                     unsigned int local_rows,
                                     unsigned int cols)
{
    int count = 0;

    for (unsigned int local_i = 0; local_i < local_rows; local_i++)
    {
        unsigned int global_i = start_row + local_i;

        for (unsigned int j = 0; j < cols; j++)
        {
            if (world[global_i * cols + j] > 0.0)
            {
                cells[count].i = global_i;
                cells[count].j = j;
                count++;
            }
        }
    }

    return count;
}

void generate_mask_from_cells(unsigned char *active,
                              unsigned int rows,
                              unsigned int cols,
                              const Cell *cells,
                              int cell_count,
                              const Offset *offsets,
                              int offset_count)
{
    for (int c = 0; c < cell_count; c++)
    {
        unsigned int i = cells[c].i;
        unsigned int j = cells[c].j;

        for (int k = 0; k < offset_count; k++)
        {
            int ii = (int)i + offsets[k].di;
            int jj = (int)j + offsets[k].dj;

            if (ii < 0) ii += rows;
            else if (ii >= (int)rows) ii -= rows;

            if (jj < 0) jj += cols;
            else if (jj >= (int)cols) jj -= cols;

            active[ii * cols + jj] = 1;
        }
    }
}

void generate_mask_contribution_from_cells(unsigned char *mask_contribution,
                                           unsigned int rows,
                                           unsigned int cols,
                                           const Cell *cells,
                                           int cell_count,
                                           const Offset *offsets,
                                           int offset_count)
{
    generate_mask_from_cells(
        mask_contribution,
        rows,
        cols,
        cells,
        cell_count,
        offsets,
        offset_count
    );
}

static void exchange_halo_rows_nonblocking(
    double *restrict local_world,
    unsigned int local_rows,
    unsigned int cols,
    unsigned int halo,
    int rank,
    int size
)
{
    if (halo == 0)
    {
        return;
    }


    const int count = (int)((size_t)halo * (size_t)cols);

    double *top_halo =
        local_world;

    double *first_real_rows =
        local_world + (size_t)halo * (size_t)cols;

    double *last_real_rows =
        local_world + (size_t)local_rows * (size_t)cols;

    double *bottom_halo =
        local_world + ((size_t)halo + (size_t)local_rows) * (size_t)cols;

    if (size == 1)
    {
        memcpy(
            top_halo,
            last_real_rows,
            (size_t)count * sizeof(double)
        );

        memcpy(
            bottom_halo,
            first_real_rows,
            (size_t)count * sizeof(double)
        );

        return;
    }

    int up_rank = rank - 1;
    int down_rank = rank + 1;

    if (up_rank < 0)
    {
        up_rank = size - 1;
    }

    if (down_rank >= size)
    {
        down_rank = 0;
    }

    MPI_Request requests[4];

    MPI_Irecv(
        top_halo,
        count,
        MPI_DOUBLE,
        up_rank,
        HALO_TAG_DOWN,
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Irecv(
        bottom_halo,
        count,
        MPI_DOUBLE,
        down_rank,
        HALO_TAG_UP,
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Isend(
        first_real_rows,
        count,
        MPI_DOUBLE,
        up_rank,
        HALO_TAG_UP,
        MPI_COMM_WORLD,
        &requests[2]
    );

    MPI_Isend(
        last_real_rows,
        count,
        MPI_DOUBLE,
        down_rank,
        HALO_TAG_DOWN,
        MPI_COMM_WORLD,
        &requests[3]
    );

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

static void generate_active_mask_local_scatter_from_halo(
    unsigned char *restrict active_local,
    const double *restrict local_world,
    const unsigned int local_rows,
    const unsigned int cols,
    const unsigned int halo,
    const Offset *restrict offsets,
    const int offset_count
)
{
    const unsigned int total_local_rows = local_rows + 2 * halo;

    memset(
        active_local,
        0,
        (size_t)local_rows * (size_t)cols * sizeof(*active_local)
    );

    for (unsigned int src_row = 0; src_row < total_local_rows; src_row++)
    {
        const double *restrict world_row =
            local_world + (size_t)src_row * (size_t)cols;

        for (unsigned int src_col = 0; src_col < cols; src_col++)
        {
            if (world_row[src_col] <= 0.0)
            {
                continue;
            }

            for (int k = 0; k < offset_count; k++)
            {
                int dst_row_with_halo = (int)src_row + offsets[k].di;

                if (dst_row_with_halo < (int)halo ||
                    dst_row_with_halo >= (int)(halo + local_rows))
                {
                    continue;
                }

                unsigned int dst_local_row =
                    (unsigned int)(dst_row_with_halo - (int)halo);

                int dst_col = (int)src_col + offsets[k].dj;

                if (dst_col < 0)
                {
                    dst_col += (int)cols;
                }
                else if (dst_col >= (int)cols)
                {
                    dst_col -= (int)cols;
                }

                active_local[
                    (size_t)dst_local_row * (size_t)cols + (unsigned int)dst_col
                ] = 1;
            }
        }
    }
}

static void generate_active_mask_local_scatter_from_full_world(
    unsigned char *restrict active_local,
    const double *restrict world,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int start_row,
    const unsigned int local_rows,
    const Offset *restrict offsets,
    const int offset_count
)
{
    if (active_local == NULL || world == NULL || offsets == NULL)
    {
        return;
    }

    memset(
        active_local,
        0,
        (size_t)local_rows * (size_t)cols * sizeof(*active_local)
    );

    const unsigned int end_row = start_row + local_rows;

    for (unsigned int src_row = 0; src_row < rows; src_row++)
    {
        const double *restrict world_row =
            world + (size_t)src_row * (size_t)cols;

        for (unsigned int src_col = 0; src_col < cols; src_col++)
        {
            if (world_row[src_col] <= 0.0)
            {
                continue;
            }

            for (int k = 0; k < offset_count; k++)
            {
                unsigned int dst_global_row = wrap_unsigned_index(
                    (int)src_row + offsets[k].di,
                    rows
                );

                if (dst_global_row < start_row || dst_global_row >= end_row)
                {
                    continue;
                }

                unsigned int dst_local_row = dst_global_row - start_row;

                unsigned int dst_col = wrap_unsigned_index(
                    (int)src_col + offsets[k].dj,
                    cols
                );

                active_local[
                    (size_t)dst_local_row * (size_t)cols + (size_t)dst_col
                ] = 1;
            }
        }
    }
}

static double *convolve2d_rows_local_halo(
    double *restrict local_result,
    const double *restrict local_world,
    const double *restrict w,
    const unsigned int local_rows,
    const unsigned int cols,
    const unsigned int halo,
    const unsigned int w_rows,
    const unsigned int w_cols,
    const unsigned char *restrict active_local
)
{
    if (local_result == NULL ||
        local_world == NULL ||
        w == NULL ||
        active_local == NULL)
    {
        return local_result;
    }

    const int row_center = (int)w_rows / 2;
    const int col_center = (int)w_cols / 2;

    for (unsigned int local_row = 0; local_row < local_rows; local_row++)
    {
        double *restrict result_row =
            local_result + (size_t)local_row * (size_t)cols;

        const unsigned char *restrict active_row =
            active_local + (size_t)local_row * (size_t)cols;

        const unsigned int real_row = halo + local_row;

        for (unsigned int col = 0; col < cols; col++)
        {
            if (!active_row[col])
            {
                result_row[col] = 0.0;
                continue;
            }

            double sum = 0.0;

            for (unsigned int kr = 0; kr < w_rows; kr++)
            {
                const int src_row =
                    (int)real_row - row_center + (int)kr;

                const double *restrict world_row =
                    local_world + (size_t)src_row * (size_t)cols;

                const double *restrict kernel_row =
                    w + (size_t)kr * (size_t)w_cols;

                for (unsigned int kc = 0; kc < w_cols; kc++)
                {
                    unsigned int src_col = wrap_unsigned_index(
                        (int)col - col_center + (int)kc,
                        cols
                    );

                    sum += kernel_row[kc] * world_row[src_col];
                }
            }

            result_row[col] = sum;
        }
    }

    return local_result;
}

// Function to evolve Lenia
double *evolve_lenia(
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int steps,
    const double dt,
    const unsigned int kernel_size,
    const struct orbium_coo *orbiums,
    const unsigned int num_orbiums,
    const int rank,
    const int size
)
{
    unsigned int sim_rows = rows;
    unsigned int sim_cols = cols;
    unsigned int sim_steps = steps;
    unsigned int sim_kernel_size = kernel_size;
    double sim_dt = dt;

    MPI_Bcast(&sim_rows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sim_cols, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sim_steps, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sim_kernel_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sim_dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (sim_rows == 0 || sim_cols == 0 || sim_kernel_size == 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Error: rows, cols and kernel_size must be > 0.\n");
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (sim_rows % (unsigned int)size != 0)
    {
        if (rank == 0)
        {
            fprintf(
                stderr,
                "Error: number of rows (%u) must be divisible by number of MPI ranks (%d).\n"
                "For non-divisible row counts, use MPI_Scatterv/Gatherv instead.\n",
                sim_rows,
                size
            );
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const unsigned int halo = sim_kernel_size / 2;

    size_t total_count_size =
        (size_t)sim_rows * (size_t)sim_cols;

    size_t local_rows_size =
        (size_t)sim_rows / (size_t)size;

    size_t local_count_size =
        local_rows_size * (size_t)sim_cols;

    size_t halo_count_size =
        (size_t)halo * (size_t)sim_cols;

    size_t local_with_halo_rows_size =
        local_rows_size + 2 * (size_t)halo;

    size_t local_with_halo_count_size =
        local_with_halo_rows_size * (size_t)sim_cols;

    const int use_full_world_fallback =
        (local_rows_size < (size_t)halo) ? 1 : 0;

    if (use_full_world_fallback && rank == 0)
    {
        fprintf(
            stderr,
            "Info: local_rows (%zu) is smaller than halo size (%u).\n"
            "Using full-world MPI_Allgather fallback for these parameters.\n",
            local_rows_size,
            halo
        );
    }

    if (total_count_size > (size_t)INT_MAX ||
        local_count_size > (size_t)INT_MAX ||
        halo_count_size > (size_t)INT_MAX)
    {
        if (rank == 0)
        {
            fprintf(
                stderr,
                "Error: MPI count is larger than INT_MAX. Use a larger-count MPI strategy.\n"
            );
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unsigned int local_rows = (unsigned int)local_rows_size;
    int local_count = (int)local_count_size;

#ifdef GENERATE_GIF
    ge_GIF *gif = NULL;

    if (rank == 0)
    {
        gif = ge_new_gif(
            "lenia.gif",
            sim_cols,
            sim_rows,
            inferno_pallete,
            8,
            -1,
            0
        );

        if (gif == NULL)
        {
            fprintf(stderr, "Rank 0: failed to create GIF.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
#endif

    double *w = (double *)calloc(
        (size_t)sim_kernel_size * (size_t)sim_kernel_size,
        sizeof(double)
    );


    double *world = (double *)calloc(
        total_count_size,
        sizeof(double)
    );

    double *local_world = (double *)calloc(
        local_with_halo_count_size,
        sizeof(double)
    );

    double *local_next = (double *)calloc(
        local_with_halo_count_size,
        sizeof(double)
    );

    double *local_tmp = (double *)calloc(
        local_count_size,
        sizeof(double)
    );

    unsigned char *active_local = (unsigned char *)calloc(
        local_count_size,
        sizeof(unsigned char)
    );

    Offset *offsets = (Offset *)malloc(
        (size_t)sim_kernel_size * (size_t)sim_kernel_size * sizeof(Offset)
    );

    if (w == NULL ||
        world == NULL ||
        local_world == NULL ||
        local_next == NULL ||
        local_tmp == NULL ||
        active_local == NULL ||
        offsets == NULL)
    {
        fprintf(stderr, "Rank %d: failed to allocate simulation arrays.\n", rank);

        free(w);
        free(world);
        free(local_world);
        free(local_next);
        free(local_tmp);
        free(active_local);
        free(offsets);

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    generate_kernel(w, sim_kernel_size);
    int offset_count = build_circle_offsets(offsets, sim_kernel_size);

    if (rank == 0)
    {
        for (unsigned int o = 0; o < num_orbiums; o++)
        {
            world = place_orbium(
                world,
                sim_rows,
                sim_cols,
                orbiums[o].row,
                orbiums[o].col,
                orbiums[o].angle
            );

            if (world == NULL)
            {
                fprintf(stderr, "Rank 0: place_orbium returned NULL.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    double *local_real =
        local_world + (size_t)halo * (size_t)sim_cols;

    MPI_Scatter(
        world,
        local_count,
        MPI_DOUBLE,
        local_real,
        local_count,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (use_full_world_fallback)
    {
        MPI_Allgather(
            local_real,
            local_count,
            MPI_DOUBLE,
            world,
            local_count,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );
    }

#ifdef GENERATE_GIF
    if (rank == 0)
    {
        for (unsigned int i = 0; i < sim_rows * sim_cols; i++)
        {
            gif->frame[i] = (uint8_t)(world[i] * 255.0);
        }

        ge_add_frame(gif, 5);
    }
#endif

    for (unsigned int step = 0; step < sim_steps; step++)
    {
        const unsigned int start_row = (unsigned int)rank * local_rows;

        if (use_full_world_fallback)
        {
            generate_active_mask_local_scatter_from_full_world(
                active_local,
                world,
                sim_rows,
                sim_cols,
                start_row,
                local_rows,
                offsets,
                offset_count
            );

            convolve2d_rows_local_result(
                local_tmp,
                world,
                w,
                sim_rows,
                sim_cols,
                sim_kernel_size,
                sim_kernel_size,
                start_row,
                local_rows,
                active_local
            );
        }
        else
        {

            exchange_halo_rows_nonblocking(
                local_world,
                local_rows,
                sim_cols,
                halo,
                rank,
                size
            );

            generate_active_mask_local_scatter_from_halo(
                active_local,
                local_world,
                local_rows,
                sim_cols,
                halo,
                offsets,
                offset_count
            );

            convolve2d_rows_local_halo(
                local_tmp,
                local_world,
                w,
                local_rows,
                sim_cols,
                halo,
                sim_kernel_size,
                sim_kernel_size,
                active_local
            );
        }

        local_real =
            local_world + (size_t)halo * (size_t)sim_cols;

        double *restrict local_next_real =
            local_next + (size_t)halo * (size_t)sim_cols;

        for (int i = 0; i < local_count; i++)
        {
            double value = local_real[i] + sim_dt * growth_lenia(local_tmp[i]);

            if (value < 0.0)
            {
                value = 0.0;
            }
            else if (value > 1.0)
            {
                value = 1.0;
            }

            local_next_real[i] = value;
        }

        double *swap_tmp = local_world;
        local_world = local_next;
        local_next = swap_tmp;

        local_real =
            local_world + (size_t)halo * (size_t)sim_cols;

#ifdef GENERATE_GIF
        if (use_full_world_fallback)
        {
            MPI_Allgather(
                local_real,
                local_count,
                MPI_DOUBLE,
                world,
                local_count,
                MPI_DOUBLE,
                MPI_COMM_WORLD
            );
        }
        else
        {
            MPI_Gather(
                local_real,
                local_count,
                MPI_DOUBLE,
                world,
                local_count,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD
            );
        }

        if (rank == 0)
        {
            for (unsigned int i = 0; i < sim_rows * sim_cols; i++)
            {
                gif->frame[i] = (uint8_t)(world[i] * 255.0);
            }

            ge_add_frame(gif, 5);
        }
#else
        if (use_full_world_fallback)
        {
            MPI_Allgather(
                local_real,
                local_count,
                MPI_DOUBLE,
                world,
                local_count,
                MPI_DOUBLE,
                MPI_COMM_WORLD
            );
        }
#endif
    }

    local_real =
        local_world + (size_t)halo * (size_t)sim_cols;

    MPI_Gather(
        local_real,
        local_count,
        MPI_DOUBLE,
        world,
        local_count,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

#ifdef GENERATE_GIF
    if (rank == 0)
    {
        ge_close_gif(gif);
    }
#endif

    free(w);
    free(local_world);
    free(local_next);
    free(local_tmp);
    free(active_local);
    free(offsets);

    return world;
}