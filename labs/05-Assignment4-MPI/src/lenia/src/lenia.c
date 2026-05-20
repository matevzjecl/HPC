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

/*
    Uncomment this only for small visual tests.
    For performance benchmarks, keep it disabled because GIF writing dominates runtime.
*/
// #define GENERATE_GIF

// Function to calculate Gaussian
static inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma);
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    double mu = 0.5;
    double sigma = 0.15;
    int r = (int)size / 2;
    double sum = 0.0;

    if (K == NULL)
    {
        return K;
    }

    for (unsigned int y = 0; y < size; y++)
    {
        for (unsigned int x = 0; x < size; x++)
        {
            int dy = (int)y - r;
            int dx = (int)x - r;

            double distance = sqrt((double)(dx * dx + dy * dy)) / (double)r;

            K[y * size + x] = gauss(distance, mu, sigma);

            if (distance > 1.0)
            {
                K[y * size + x] = 0.0;
            }

            sum += K[y * size + x];
        }
    }

    if (sum != 0.0)
    {
        for (unsigned int y = 0; y < size; y++)
        {
            for (unsigned int x = 0; x < size; x++)
            {
                K[y * size + x] /= sum;
            }
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
    double *result,
    const double *input,
    const double *w,
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

/*
    Row-based convolution for MPI.

    Each rank has the full current world, but only computes its own horizontal
    stripe. active_local is also local and dense:

        active_local[0]                     -> global row start_row
        active_local[cols]                  -> global row start_row + 1
        ...
        active_local[(local_rows - 1)*cols] -> global row start_row + local_rows - 1
*/
static double *convolve2d_rows_local_result(
    double *local_result,
    const double *input,
    const double *w,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int w_rows,
    const unsigned int w_cols,
    const unsigned int start_row,
    const unsigned int local_rows,
    const unsigned char *active_local
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

typedef struct {
    int di;
    int dj;
} Offset;

typedef struct {
    unsigned int i;
    unsigned int j;
} Cell;

int build_circle_offsets(Offset* offsets, unsigned int kernel_size)
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

/*
    Old gather-style implementation.
    It is safe without atomics because every output cell is written once.
    It can be slower when world is sparse because every output cell scans a neighborhood.
*/
void generate_mask_gather(const double* world,
                          unsigned char* active,
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

/*
    Old scatter-style implementation.
    Caller must clear active before calling this function.
*/
void generate_mask_scatter(const double* world,
                           unsigned char* active,
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

/*
    Faster scatter using precomputed circle offsets.
    Caller must clear active before calling this function.
*/
void generate_mask_scatter_fast(const double* world,
                                unsigned char* active,
                                unsigned int rows,
                                unsigned int cols,
                                const Offset* offsets,
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

int collect_nonzero_cells(const double* world,
                          Cell* cells,
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

int collect_nonzero_cells_local_rows(const double* world,
                                     Cell* cells,
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

/*
    Cell-list mask generation for a full local process.
    Caller must clear active before calling this function.
*/
void generate_mask_from_cells(unsigned char* active,
                              unsigned int rows,
                              unsigned int cols,
                              const Cell* cells,
                              int cell_count,
                              const Offset* offsets,
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

/*
    MPI mask contribution generation.

    Each rank only collects nonzero cells from its own rows. However, those cells
    can activate mask cells that belong to other ranks. Therefore each rank first
    creates a full-size contribution array. Then MPI_Reduce_scatter_block with
    MPI_MAX combines all contributions and gives each rank only its own mask rows.
*/
void generate_mask_contribution_from_cells(unsigned char* mask_contribution,
                                           unsigned int rows,
                                           unsigned int cols,
                                           const Cell* cells,
                                           int cell_count,
                                           const Offset* offsets,
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
    /*
        Do not trust rank/size passed from main.
        Use the actual MPI communicator values.
    */
    (void)rank;
    (void)size;

    int mpi_rank;
    int mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    /*
        Make sure every rank uses exactly the same simulation parameters.
        Rank 0 is the source of truth.
    */
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
        if (mpi_rank == 0)
        {
            fprintf(stderr, "Error: rows, cols and kernel_size must be > 0.\n");
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (sim_rows % (unsigned int)mpi_size != 0)
    {
        if (mpi_rank == 0)
        {
            fprintf(
                stderr,
                "Error: number of rows (%u) must be divisible by number of MPI ranks (%d).\n"
                "For non-divisible row counts, use MPI_Allgatherv/Gatherv instead.\n",
                sim_rows,
                mpi_size
            );
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t total_count_size = (size_t)sim_rows * (size_t)sim_cols;
    size_t local_rows_size = (size_t)sim_rows / (size_t)mpi_size;
    size_t local_count_size = local_rows_size * (size_t)sim_cols;

    if (total_count_size > (size_t)INT_MAX || local_count_size > (size_t)INT_MAX)
    {
        if (mpi_rank == 0)
        {
            fprintf(
                stderr,
                "Error: MPI count is larger than INT_MAX. Use a larger-count MPI strategy.\n"
            );
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unsigned int local_rows = (unsigned int)local_rows_size;
    unsigned int start_row = (unsigned int)mpi_rank * local_rows;
    int total_count = (int)total_count_size;
    int local_count = (int)local_count_size;
    int start_i = (int)((size_t)start_row * (size_t)sim_cols);

#ifdef GENERATE_GIF
    ge_GIF *gif = NULL;

    if (mpi_rank == 0)
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

    /*
        Every rank stores the full world.
        This is why MPI_Allgather is used after each step.
    */
    double *world = (double *)calloc(
        total_count_size,
        sizeof(double)
    );

    /*
        local_tmp stores convolution values for this rank's rows.
        local_next stores the updated row block that will be all-gathered.
    */
    double *local_tmp = (double *)calloc(
        local_count_size,
        sizeof(double)
    );

    double *local_next = (double *)calloc(
        local_count_size,
        sizeof(double)
    );

    /*
        Each rank builds a full-size contribution from only its local source rows.
        MPI_Reduce_scatter_block combines all contributions and returns only the
        local mask rows into active_local.
    */
    unsigned char *mask_contribution = (unsigned char *)calloc(
        total_count_size,
        sizeof(unsigned char)
    );

    unsigned char *active_local = (unsigned char *)calloc(
        local_count_size,
        sizeof(unsigned char)
    );

    Offset *offsets = (Offset *)malloc(
        (size_t)sim_kernel_size * (size_t)sim_kernel_size * sizeof(Offset)
    );

    Cell *cells = (Cell *)malloc(
        local_count_size * sizeof(Cell)
    );

    if (w == NULL ||
        world == NULL ||
        local_tmp == NULL ||
        local_next == NULL ||
        mask_contribution == NULL ||
        active_local == NULL ||
        offsets == NULL ||
        cells == NULL)
    {
        fprintf(stderr, "Rank %d: failed to allocate simulation arrays.\n", mpi_rank);

        free(w);
        free(world);
        free(local_tmp);
        free(local_next);
        free(mask_contribution);
        free(active_local);
        free(offsets);
        free(cells);

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    generate_kernel(w, sim_kernel_size);
    int offset_count = build_circle_offsets(offsets, sim_kernel_size);

    if (mpi_rank == 0)
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

    /*
        Initial full world is needed on every rank before the first convolution.
    */
    MPI_Bcast(
        world,
        total_count,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

#ifdef GENERATE_GIF
    if (mpi_rank == 0)
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
        memset(
            mask_contribution,
            0,
            total_count_size * sizeof(*mask_contribution)
        );

        int cell_count = collect_nonzero_cells_local_rows(
            world,
            cells,
            start_row,
            local_rows,
            sim_cols
        );

        generate_mask_contribution_from_cells(
            mask_contribution,
            sim_rows,
            sim_cols,
            cells,
            cell_count,
            offsets,
            offset_count
        );

        /*
            Combine all rank contributions with OR-like behavior.
            Since values are 0/1, MPI_MAX works like logical OR.
            Each rank receives only its own contiguous row block of the mask.
        */
        MPI_Reduce_scatter_block(
            mask_contribution,
            active_local,
            local_count,
            MPI_UNSIGNED_CHAR,
            MPI_MAX,
            MPI_COMM_WORLD
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

        for (int i = 0; i < local_count; i++)
        {
            double value = world[start_i + i] + sim_dt * growth_lenia(local_tmp[i]);
            local_next[i] = fmin(1.0, fmax(0.0, value));
        }

        /*
            Equal row blocks, so MPI_Allgather is enough.
            After this call, every rank again has the full updated world.
        */
        MPI_Allgather(
            local_next,
            local_count,
            MPI_DOUBLE,
            world,
            local_count,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

#ifdef GENERATE_GIF
        if (mpi_rank == 0)
        {
            for (unsigned int i = 0; i < sim_rows * sim_cols; i++)
            {
                gif->frame[i] = (uint8_t)(world[i] * 255.0);
            }

            ge_add_frame(gif, 5);
        }
#endif
    }

#ifdef GENERATE_GIF
    if (mpi_rank == 0)
    {
        ge_close_gif(gif);
    }
#endif

    free(w);
    free(local_tmp);
    free(local_next);
    free(mask_contribution);
    free(active_local);
    free(offsets);
    free(cells);

    return world;
}