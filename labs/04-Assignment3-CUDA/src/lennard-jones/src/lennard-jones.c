#define __USE_MISC
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gifenc.h"
#include "lennard-jones.h"
#include <omp.h>

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

    // const double r_list = R_CUT + r_skin;

    // const double area_list = M_PI * r_list * r_list;

    // const double expected_neighbors = density * area_list;
    // const double safety_factor = 2.0; 
    // int max_neighbors = (int)(expected_neighbors * safety_factor);


    // if (max_neighbors < 20) {
    //     max_neighbors = 20;
    // }

    c.max_neighbors = n - 1;



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
    const double *x,
    const double *y,
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

double compute_ke(const double *vx, const double *vy, unsigned int n) {
    double ke = 0.0;

    #pragma omp parallel for reduction(+:ke) if (n >= PARTICLE_LIMIT)
    for (unsigned int i = 0; i < n; ++i) {
        ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
    }

    return ke;
}

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

static void sort_particles_by_cell(
    double *x,
    double *y,
    double *vx,
    double *vy,
    unsigned int n,
    int *particle_cell_indices,
    int *particle_ids,
    int *cell_start,
    int *next_pos,
    double *temp_x,
    double *temp_y,
    double *temp_vx,
    double *temp_vy,
    const constants *c
) {
    const int num_cells = c->num_cells;
    const int num_cells_x = c->num_cells_x;
    const int num_cells_y = c->num_cells_y;

    const double inv_cell_size_x = c->inv_cell_size_x;
    const double inv_cell_size_y = c->inv_cell_size_y;

    memset(cell_start, 0, (size_t)(num_cells + 1) * sizeof(*cell_start));

    for (unsigned int i = 0; i < n; ++i) {
        int cell_x = (int)(x[i] * inv_cell_size_x);
        int cell_y = (int)(y[i] * inv_cell_size_y);

        if (cell_x >= num_cells_x) cell_x = num_cells_x - 1;
        if (cell_y >= num_cells_y) cell_y = num_cells_y - 1;
        if (cell_x < 0) cell_x = 0;
        if (cell_y < 0) cell_y = 0;

        const int cell = cell_y * num_cells_x + cell_x;

        particle_cell_indices[i] = cell;
        cell_start[cell + 1]++;
    }

    for (int cell = 0; cell < num_cells; ++cell) {
        cell_start[cell + 1] += cell_start[cell];
    }

    memcpy(next_pos, cell_start, (size_t)num_cells * sizeof(*next_pos));

    for (unsigned int i = 0; i < n; ++i) {
        const int cell = particle_cell_indices[i];
        const int pos = next_pos[cell]++;

        particle_ids[pos] = (int)i;
    }

    #pragma omp parallel for if (n >= PARTICLE_LIMIT)
    for (unsigned int i = 0; i < n; ++i) {
        const int old_id = particle_ids[i];

        temp_x[i] = x[old_id];
        temp_y[i] = y[old_id];
        temp_vx[i] = vx[old_id];
        temp_vy[i] = vy[old_id];
    }
    memcpy(x, temp_x, (size_t)n * sizeof(*x));
    memcpy(y, temp_y, (size_t)n * sizeof(*y));
    memcpy(vx, temp_vx, (size_t)n * sizeof(*vx));
    memcpy(vy, temp_vy, (size_t)n * sizeof(*vy));
}

static void build_verlet_list(
    const double *x,
    const double *y,
    unsigned int n,
    double local_box_size,
    double inv_box_size,
    double r_skin,
    int *neighbor_counts,
    int *neighbor_list,
    const int max_neighbors
) {
    const double r_list = R_CUT + r_skin;
    const double r_list2 = r_list * r_list;

    memset(neighbor_counts, 0, (size_t)n * sizeof(*neighbor_counts));

    #pragma omp parallel for
    for (unsigned int i = 0; i < n; ++i) {
        int count = 0;
        const int list_offset = (int)i * max_neighbors;

        for (unsigned int j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }

            double dx = x[i] - x[j];
            double dy = y[i] - y[j];

            dx -= local_box_size * nearbyint(dx * inv_box_size);
            dy -= local_box_size * nearbyint(dy * inv_box_size);

            const double dist2 = dx * dx + dy * dy;

            if (dist2 < r_list2) {
                if (count < max_neighbors) {
                    neighbor_list[list_offset + count] = (int)j;
                    count++;
                } else {
                    printf("WARNING: MAX_NEIGHBORS exceeded for particle %u\n", i);
                }
            }
        }

        neighbor_counts[i] = count;
    }
}

void build_cells(
    const double *x,
    const double *y,
    unsigned int n,
    int *head,
    int *next,
    const constants *c
) {
    const int num_cells = c->num_cells;
    const int num_cells_x = c->num_cells_x;
    const int num_cells_y = c->num_cells_y;

    const double inv_cell_size_x = c->inv_cell_size_x;
    const double inv_cell_size_y = c->inv_cell_size_y;

    for (int cell = 0; cell < num_cells; cell++) {
        head[cell] = -1;
    }
    for (unsigned int i = 0; i < n; i++) {
        int cell_x = (int)(x[i] * inv_cell_size_x);
        int cell_y = (int)(y[i] * inv_cell_size_y);

        if (cell_x >= num_cells_x) cell_x = num_cells_x - 1;
        if (cell_y >= num_cells_y) cell_y = num_cells_y - 1;
        if (cell_x < 0) cell_x = 0;
        if (cell_y < 0) cell_y = 0;

        const int cell = cell_y * num_cells_x + cell_x;

        next[i] = head[cell];
        head[cell] = (int)i;
    }
}

void build_verlet_list_cells(
    const double *x, 
    const double *y, 
    unsigned int n,
    int *neighbor_counts,
    int *neighbor_list,
    int *head,
    int *next,
    double r_skin,
    const constants *c
) {
    const double local_box_size = c->box_size;
    const double inv_box_size = c->inv_box_size;
    
    const double r_list = R_CUT + r_skin;
    const double r_list2 = r_list * r_list;

    const int num_cells_x = c->num_cells_x;
    const int num_cells_y = c->num_cells_y;

    memset(neighbor_counts, 0, n * sizeof(int));

    build_cells(x, y, n, head, next, c);

    // #pragma omp parallel for schedule(dynamic, 32)
    for (int cy = 0; cy < num_cells_y; cy++) {
        for (int cx = 0; cx < num_cells_x; cx++) {
            const int cell = cy * num_cells_x + cx;

            for (int i = head[cell]; i != -1; i = next[i]) {
                int count = 0;
                const int list_offset = i * c->max_neighbors;

                for (int oy = -1; oy <= 1; oy++) {
                    for (int ox = -1; ox <= 1; ox++) {
                        int ncx = cx + ox;
                        int ncy = cy + oy;

                        if (ncx < 0) ncx += num_cells_x;
                        else if (ncx >= num_cells_x) ncx -= num_cells_x;

                        if (ncy < 0) ncy += num_cells_y;
                        else if (ncy >= num_cells_y) ncy -= num_cells_y;

                        const int neighbor_cell = ncy * num_cells_x + ncx;

                        for (int j = head[neighbor_cell]; j != -1; j = next[j]) {
                            if (i == j) continue;

                            double dx = x[i] - x[j];
                            double dy = y[i] - y[j];

                            dx -= local_box_size * nearbyint(dx * inv_box_size);
                            dy -= local_box_size * nearbyint(dy * inv_box_size);

                            const double dist2 = dx * dx + dy * dy;

                            if (dist2 < r_list2) {
                                neighbor_list[list_offset + count] = j;
                                count++;
                            }
                        }
                    }
                }
                neighbor_counts[i] = count;
            }
        }
    }
}

static void build_cell_neighbors_full(
    int *cell_neighbors,
    const constants *c
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

static void build_verlet_list_from_sorted_cells(
    const double *x,
    const double *y,
    unsigned int n,
    const int *cell_start,
    const int *cell_neighbors,
    int *neighbor_counts,
    int *neighbor_list,
    float r_skin,
    const constants *c
) {
    const double local_box_size = c->box_size;
    const double inv_box_size = c->inv_box_size;

    const double r_list = (double)R_CUT + (double)r_skin;
    const double r_list2 = r_list * r_list;

    const int num_cells = c->num_cells;
    const int max_neighbors = c->max_neighbors;

    #pragma omp parallel for schedule(static)
    for (int cell_i = 0; cell_i < num_cells; ++cell_i) {
        const int i_start = cell_start[cell_i];
        const int i_end = cell_start[cell_i + 1];

        for (int i = i_start; i < i_end; ++i) {
            int count = 0;
            const int list_offset = i * max_neighbors;

            for (int k = 0; k < NEIGHBOR_COUNT; ++k) {
                const int cell_j = cell_neighbors[cell_i * NEIGHBOR_COUNT + k];

                const int j_start = cell_start[cell_j];
                const int j_end = cell_start[cell_j + 1];

                for (int j = j_start; j < j_end; ++j) {
                    if (j == i) {
                        continue;
                    }

                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];

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
    }
}

static double compute_forces_verlet(
    const double *x,
    const double *y,
    double *fx,
    double *fy,
    unsigned int n,
    double local_box_size,
    double inv_box_size,
    const int *neighbor_counts,
    const int *neighbor_list,
    const int max_neighbors
) {
    const double r_cut2 = R_CUT2;
    const double sigma2 = SIGMA2;
    const double v_shift = V_SHIFT;
    const double four_epsilon = FOUR_EPSILON;
    const double twentyfour_epsilon = TWENTYFOUR_EPSILON;

    double total_pe = 0.0;

    #pragma omp parallel for
    for (unsigned int i = 0; i < n; ++i) {
        fx[i] = 0.0;
        fy[i] = 0.0;
    }

    #pragma omp parallel for reduction(+:total_pe)
    for (unsigned int i = 0; i < n; ++i) {
        const int count = neighbor_counts[i];
        const int list_offset = (int)i * max_neighbors;

        double fxi = 0.0;
        double fyi = 0.0;
        double pei = 0.0;

        for (int k = 0; k < count; ++k) {
            const int j = neighbor_list[list_offset + k];

            double dx = x[i] - x[j];
            double dy = y[i] - y[j];

            dx -= local_box_size * nearbyint(dx * inv_box_size);
            dy -= local_box_size * nearbyint(dy * inv_box_size);

            const double r2 = dx * dx + dy * dy;

            if (r2 >= r_cut2 || r2 == 0.0) {
                continue;
            }

            const double inv_r2 = 1.0 / r2;
            const double sr2 = sigma2 * inv_r2;
            const double sr6 = sr2 * sr2 * sr2;
            const double sr12 = sr6 * sr6;

            const double fij = twentyfour_epsilon * (2.0 * sr12 - sr6) * inv_r2;

            fxi += fij * dx;
            fyi += fij * dy;

            const double vij = four_epsilon * (sr12 - sr6) - v_shift;

            pei += 0.5 * vij;
        }

        fx[i] = fxi;
        fy[i] = fyi;
        total_pe += pei;
    }

    return total_pe;
}

static double leapfrog_step(
    double *x,
    double *y,
    double *vx,
    double *vy,
    double *fx,
    double *fy,
    const unsigned int n,
    double *ke_out,
    double *temp_x,
    double *temp_y,
    double *temp_vx,
    double *temp_vy,
    int *particle_cell_indices,
    int *particle_ids,
    int *cell_start,
    int *next_pos,
    int *needs_rebuild,
    int *neighbor_counts,
    int *neighbor_list,
    float r_skin,
    const int *cell_neighbors,
    const constants *c
) {
    const double local_box_size = c->box_size;
    const double inv_box_size = c->inv_box_size;
    const double max_disp2 = c->max_disp2;

    int rebuild_local = 0;

    #pragma omp parallel for reduction(|:rebuild_local)
    for (unsigned int i = 0; i < n; ++i) {
        vx[i] += HALF_DT * fx[i];
        vy[i] += HALF_DT * fy[i];

        x[i] += DT * vx[i];
        y[i] += DT * vy[i];

        double wx = fmod(x[i], local_box_size);
        double wy = fmod(y[i], local_box_size);

        if (wx < 0.0) wx += local_box_size;
        if (wy < 0.0) wy += local_box_size;

        x[i] = wx;
        y[i] = wy;

        double dx = x[i] - temp_x[i];
        double dy = y[i] - temp_y[i];

        dx -= local_box_size * nearbyint(dx * inv_box_size);
        dy -= local_box_size * nearbyint(dy * inv_box_size);

        const double dist2 = dx * dx + dy * dy;

        if (dist2 > max_disp2) {
            rebuild_local = 1;
        }
    }

    if (rebuild_local) {
        *needs_rebuild = 1;
    }

    if (*needs_rebuild) {
        sort_particles_by_cell(
            x,
            y,
            vx,
            vy,
            n,
            particle_cell_indices,
            particle_ids,
            cell_start,
            next_pos,
            temp_x,
            temp_y,
            temp_vx,
            temp_vy,
            c
        );

        build_verlet_list_from_sorted_cells(
            x,
            y,
            n,
            cell_start,
            cell_neighbors,
            neighbor_counts,
            neighbor_list,
            r_skin,
            c
        );

        *needs_rebuild = 0;
    }

    const double pe = compute_forces_verlet(
        x,
        y,
        fx,
        fy,
        n,
        local_box_size,
        inv_box_size,
        neighbor_counts,
        neighbor_list,
        c->max_neighbors
    );

    double ke = 0.0;

    #pragma omp parallel for reduction(+:ke) if (n >= PARTICLE_LIMIT)
    for (unsigned int i = 0; i < n; ++i) {
        vx[i] += HALF_DT * fx[i];
        vy[i] += HALF_DT * fy[i];

        ke += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
    }

    *ke_out = ke;

    return pe;
}

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
) {
    (void)block_size;
    (void)density;
    SimulationResult out;

    const constants c = make_constants(box_size, r_skin, n);

    int *neighbor_counts = malloc((size_t)n * sizeof(*neighbor_counts));
    int *neighbor_list = malloc((size_t)n * (size_t)c.max_neighbors * sizeof(*neighbor_list));

    int *cell_neighbors = malloc((size_t)c.num_cells * NEIGHBOR_COUNT * sizeof(*cell_neighbors));

    double *temp_x = malloc((size_t)n * sizeof(*temp_x));
    double *temp_y = malloc((size_t)n * sizeof(*temp_y));
    double *temp_vx = malloc((size_t)n * sizeof(*temp_vx));
    double *temp_vy = malloc((size_t)n * sizeof(*temp_vy));

    int *particle_cell_indices = malloc((size_t)n * sizeof(*particle_cell_indices));
    int *particle_ids = malloc((size_t)n * sizeof(*particle_ids));

    int *cell_start = malloc((size_t)(c.num_cells + 1) * sizeof(*cell_start));
    int *next_pos = malloc((size_t)c.num_cells * sizeof(*next_pos));

    if (!neighbor_counts || !neighbor_list ||
        !temp_x || !temp_y || !temp_vx || !temp_vy ||
        !particle_cell_indices || !particle_ids ||
        !cell_start || !next_pos || !cell_neighbors) {
        fprintf(stderr, "Allocation failed\n");

        free(neighbor_counts);
        free(neighbor_list);

        free(temp_x);
        free(temp_y);
        free(temp_vx);
        free(temp_vy);

        free(particle_cell_indices);
        free(particle_ids);

        free(cell_start);
        free(next_pos);

        free(cell_neighbors);

        exit(EXIT_FAILURE);
    }


    int needs_rebuild = 1;

    build_cell_neighbors_full(cell_neighbors, &c);

    sort_particles_by_cell(
        x,
        y,
        vx,
        vy,
        n,
        particle_cell_indices,
        particle_ids,
        cell_start,
        next_pos,
        temp_x,
        temp_y,
        temp_vx,
        temp_vy,
        &c
    );

    build_verlet_list_from_sorted_cells(
        x,
        y,
        n,
        cell_start,
        cell_neighbors,
        neighbor_counts,
        neighbor_list,
        r_skin,
        &c
    );

    needs_rebuild = 0;

    out.start_potential = compute_forces_verlet(
        x,
        y,
        fx,
        fy,
        n,
        c.box_size,
        c.inv_box_size,
        neighbor_counts,
        neighbor_list,
        c.max_neighbors
    );

    out.start_kinetic = compute_ke(vx, vy, n);
    out.start_total = out.start_kinetic + out.start_potential;

    out.final_potential = out.start_potential;
    out.final_kinetic = out.start_kinetic;
    out.final_total = out.start_total;

#if GENERATE_GIF
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
        out.final_potential = leapfrog_step(
            x,
            y,
            vx,
            vy,
            fx,
            fy,
            n,
            &out.final_kinetic,
            temp_x,
            temp_y,
            temp_vx,
            temp_vy,
            particle_cell_indices,
            particle_ids,
            cell_start,
            next_pos,
            &needs_rebuild,
            neighbor_counts,
            neighbor_list,
            r_skin,
            cell_neighbors,
            &c
        );

        out.final_total = out.final_kinetic + out.final_potential;

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
            render_frame_gif(gif, x, y, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    free(neighbor_counts);
    free(neighbor_list);

    free(temp_x);
    free(temp_y);
    free(temp_vx);
    free(temp_vy);

    free(particle_cell_indices);
    free(particle_ids);

    free(cell_start);
    free(next_pos);
    free(cell_neighbors);
    out.n = n;

    return out;
}