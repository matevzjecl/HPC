#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "lennard-jones.h"

void print_help(const char *exe) {
    printf("Usage: %s [N] [nsteps] [r_skin]\n", exe);
}

int main(int argc, char **argv) {
    unsigned int nsteps = 5000;
    unsigned int n = 1000;
    double density = 0.95;
    double temperature = 0.5;
    unsigned int seed = 42;
    float r_skin = 0.4;
    int block_size = 64;

    SimulationResult result;

    if (argc > 1) {
        n = (unsigned int)strtoul(argv[1], NULL, 10);
    }

    if (argc > 2) {
        nsteps = (unsigned int)strtoul(argv[2], NULL, 10);
    }

    if (argc > 3) {
        r_skin = (float)strtof(argv[3], NULL);
    }

    if (argc > 4) {
        block_size = (int)strtol(argv[4], NULL, 10);
    }

    if (argc > 5) {
        print_help(argv[0]);
        return 1;
    }

    double particle_box_size = ceil(sqrt((double)n / density));
    double box_size = (4.0 / 3.0) * particle_box_size;
    double box_fraction = particle_box_size / box_size;

    double *x  = calloc(n, sizeof(double));
    double *y  = calloc(n, sizeof(double));
    double *vx = calloc(n, sizeof(double));
    double *vy = calloc(n, sizeof(double));
    double *fx = calloc(n, sizeof(double));
    double *fy = calloc(n, sizeof(double));

    if (!x || !y || !vx || !vy || !fx || !fy) {
        fprintf(stderr, "Failed to allocate simulation arrays.\n");

        free(x);
        free(y);
        free(vx);
        free(vy);
        free(fx);
        free(fy);

        return 1;
    }

    if (!initialize_particles(
            x,
            y,
            vx,
            vy,
            fx,
            fy,
            n,
            box_size,
            box_fraction,
            seed,
            temperature
        )) {
        fprintf(stderr, "Failed to initialize particles.\n");

        free(x);
        free(y);
        free(vx);
        free(vy);
        free(fx);
        free(fy);

        return 1;
    }

    double start = omp_get_wtime();

    result = run_simulation(
        x,
        y,
        vx,
        vy,
        fx,
        fy,
        n,
        nsteps,
        box_size,
        0,
        r_skin,
        block_size,
        density
    );

    double stop = omp_get_wtime();

    printf("\nFinished simulation.\n");
    printf("Final KE: %10.4f | delta: %+.4f\n",
           result.final_kinetic,
           result.final_kinetic - result.start_kinetic);

    printf("Final PE: %10.4f | delta: %+.4f\n",
           result.final_potential,
           result.final_potential - result.start_potential);

    printf("Final E:  %10.4f | delta: %+.4f\n",
           result.final_total,
           result.final_total - result.start_total);

    printf("Simulation time %u steps: %.3f seconds\n", nsteps, stop - start);

    free(x);
    free(y);
    free(vx);
    free(vy);
    free(fx);
    free(fy);

    return 0;
}