#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "lenia.h"

#define N 4096
#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

// Place two orbiums in the world with different angles. (y, x, angle)
// Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};

// struct orbium_coo orbiums[NUM_ORBIUMS] = {
//     {0,       N / 5,   0},
//     {N / 5,   0,      18},
//     {N / 5,   N / 5,  36},
//     {N / 5,   2*N/5,  54},
//     {N / 5,   3*N/5,  72},

//     {2*N/5,   0,      90},
//     {2*N/5,   N / 5, 108},
//     {2*N/5,   2*N/5, 126},
//     {2*N/5,   3*N/5, 144},
//     {2*N/5,   4*N/5, 162},

//     {3*N/5,   0,     180},
//     {3*N/5,   N / 5, 198},
//     {3*N/5,   2*N/5, 216},
//     {3*N/5,   3*N/5, 234},
//     {3*N/5,   4*N/5, 252},

//     {4*N/5,   0,     270},
//     {4*N/5,   N / 5, 288},
//     {4*N/5,   2*N/5, 306},
//     {4*N/5,   3*N/5, 324},
//     {4*N/5,   4*N/5, 342}
// };

int main(int argc, char **argv)
{
    int block_size = 16;   // default

    if (argc > 1) {
        block_size = atoi(argv[1]);
    }

    if (block_size <= 0 || block_size * block_size > 1024) {
        fprintf(stderr, "Invalid block size: %d\n", block_size);
        fprintf(stderr, "Use a positive value where block_size * block_size <= 1024\n");
        return 1;
    }
    printf("CUDA world size: %d x %d\n", N, N);
    printf("CUDA block size: %d x %d\n", block_size, block_size);
    double start = omp_get_wtime();
    // Run the simulation
    double *world = evolve_lenia(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS, block_size);
    double stop = omp_get_wtime();
    printf("Execution time: %.3f\n", stop - start);
    free(world);
    return 0;
}