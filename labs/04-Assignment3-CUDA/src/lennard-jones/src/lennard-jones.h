#ifndef LJ_H
#define LJ_H

#ifdef __cplusplus
extern "C" {
#endif

#define DT 0.002
#define SIGMA 1.0
#define EPSILON 1.0
#define R_CUT 2.5
#define JITTER 0.05

#define GENERATE_GIF 0
#define FRAME_WIDTH 800
#define FRAME_HEIGHT 800
#define FRAME_EVERY 5
#define FRAME_PARTICLE_RADIUS 2
#define FRAME_DELAY 3
#define GIF_FILE "simulation.gif"

typedef struct {
    unsigned int n;
    double start_kinetic;
    double start_potential;
    double start_total;
    double final_kinetic;
    double final_potential;
    double final_total;
} SimulationResult;

typedef struct {
    int n_cells;
    int num_cells;
    double cell_size;
    int *head;
    int *next;
    int *particle_cell;
} CellGrid;

typedef struct {
    double box_size;
    double inv_box_size;

    double max_disp2;

    int num_cells_x;
    int num_cells_y;
    int num_cells;

    double cell_size_x;
    double cell_size_y;
    double inv_cell_size_x;
    double inv_cell_size_y;
    unsigned int max_neighbors;
} constants;


int initialize_particles(
    double *x,
    double *y,
    double *vx,
    double *vy,
    double *fx,
    double *fy,
    unsigned int n,
    double box_size,
    double placement_fraction,
    unsigned int seed,
    double temperature
);

SimulationResult run_simulation(
    double *x,
    double *y,
    double *vx,
    double *vy,
    double *fx,
    double *fy,
    unsigned int n,
    unsigned int nsteps,
    double box_size,
    int log_steps,
    float r_skin,
    int block_size,
    double density
);

#ifdef __cplusplus
}
#endif

#endif