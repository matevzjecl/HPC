#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <sched.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "sample_code/stb_image.h"
#include "sample_code/stb_image_write.h"

#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

static inline int mini(int a, int b) {
    return (a < b) ? a : b;
}

static inline int maxi(int a, int b) {
    return (a > b) ? a : b;
}

static inline float min(float a, float b) {
    return (a < b) ? a : b;
}

static inline float min3f(float a, float b, float c) {
    float m = (a < b) ? a : b;
    return (m < c) ? m : c;
}

int getPixel(const unsigned char *image, int width, int height, int cpp, int ch, int i, int j) {
    if (i < 0) i = 0;
    else if (i >= height) i = height - 1;

    if (j < 0) j = 0;
    else if (j >= width) j = width - 1;

    return image[(i * width + j) * cpp + ch];
}

void compute_energy(const unsigned char *image, int width, int height, int cpp, float *energy, int use_parallel) {
    int energy_cpp = (cpp >= 3) ? 3 : cpp;
    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float Ea = 0.0f;

            for (int ch = 0; ch < energy_cpp; ch++) {
                int topLeft = getPixel(image, width, height, cpp, ch, i - 1, j - 1);
                int left = getPixel(image, width, height, cpp, ch, i, j - 1);
                int bottomLeft = getPixel(image, width, height, cpp, ch, i + 1, j - 1);

                int topRight = getPixel(image, width, height, cpp, ch, i - 1, j + 1);
                int right = getPixel(image, width, height, cpp, ch, i, j + 1);

                int top = getPixel(image, width, height, cpp, ch, i - 1, j);
                int bottom = getPixel(image, width, height, cpp, ch, i + 1, j);
                int bottomRight = getPixel(image, width, height, cpp, ch, i + 1, j + 1);

                int Gx = -topLeft - 2 * left - bottomLeft + topRight + 2 * right + bottomRight;
                int Gy = topLeft + 2 * top + topRight - bottomLeft - 2 * bottom - bottomRight;

                Ea += sqrtf((float)(Gx * Gx + Gy * Gy));
            }

            energy[i * width + j] = Ea / (float) energy_cpp;
        }
    }
}

void accumulate_single(float *energy, int i, int j, int width) {
    float bottom = energy[(i + 1) * width + j];
    float bottomLeft = INFINITY;
    float bottomRight = INFINITY;

    if (j > 0) {
        bottomLeft = energy[(i + 1) * width + (j - 1)];
    }
    if (j < width - 1) {
        bottomRight = energy[(i + 1) * width + (j + 1)];
    }

    energy[i * width + j] += min3f(bottomLeft, bottom, bottomRight);
}

void accumulate_triangle(float *energy, int h, int w, int width, int height,
                         int triangle_height, int triangle_width,
                         int is_half, int up) {
    int start_w = w;
    const int cell = 4;

    if (up) {
        int i_start = h;
        int i_end = maxi(h - triangle_height + 1, 0);
        for (int i = i_start; i >= i_end; i--) {
            int indent = w - start_w;
            int j_start = maxi(w, 0);
            int j_end = mini(w + triangle_width - 1, width - 1);

            for (int j = j_start; j <= j_end; j++) {
                accumulate_single(energy, i, j, width);
            }

            if (is_half == 1) {
                triangle_width--;
            } else {
                w++;
                triangle_width -= 2;
            }
        }
    } else {
        int i_start = mini(h + triangle_height - 1, height - 2);
        int i_end = maxi(h, 0);

        int row_offset = i_start - h;
        int cur_w = w + (is_half ? 0 : row_offset);
        int cur_triangle_width = triangle_width - (is_half ? row_offset : 2 * row_offset);

        for (int i = i_start; i >= i_end; i--) {
            int indent = cur_w - start_w;

            int j_start = maxi(cur_w, 0);
            int j_end = mini(cur_w + cur_triangle_width - 1, width - 1);

            for (int j = j_start; j <= j_end; j++) {
                accumulate_single(energy, i, j, width);
            }


            if (is_half == 1) {
                cur_triangle_width++;
            } else {
                cur_w--;
                cur_triangle_width += 2;
            }
        }
    }
}

void accumulate_energy(float *energy, int width, int height, int use_parallel, int use_triangles, int triangle_width) {
    int rows_to_compute = height - 1;

    if (use_triangles == 0) {
            #pragma omp parallel if(use_parallel) 
            {
                for (int i = rows_to_compute - 1; i >= 0; i--) {
                    
                    #pragma omp for schedule(static)
                    for (int j = 0; j < width; j++) {
                        accumulate_single(energy, i, j, width);
                    }
                }
            }
    } else {
        int triangle_h = triangle_width / 2;
        int triangle_n = (width + triangle_width - 1) / triangle_width;
        int strips = (rows_to_compute + triangle_h) / (triangle_h + 1);
        int current_height = height - 2;

        #pragma omp parallel if(use_parallel)
        {
            for (int s = 0; s < strips; s++) {
                if (current_height < 0) {
                    break;
                }

                int strip_height = mini(triangle_h, current_height + 1);

                #pragma omp for schedule(static)
                for (int i = 0; i < triangle_n; i++) {
                    int start_col = i * triangle_width;
                    int tri_width = triangle_width;
                    int is_half = 0;

                    accumulate_triangle(energy, current_height, start_col, width, height,
                                        strip_height, tri_width, is_half, 1);
                }

                int down_h = current_height - strip_height;

                if (down_h >= 0) {
                    int down_offset = triangle_width / 2;
                    #pragma omp for schedule(static)
                    for (int i = 0; i < triangle_n + 1; i++) {
                        int start_col;
                        int tri_width;
                        int is_half;

                        if (i == 0) {
                            start_col = 0;
                            tri_width = down_offset;
                            is_half = 1;
                        } else {
                            start_col = (i - 1) * triangle_width + down_offset;
                            tri_width = triangle_width;
                            is_half = 0;
                        }

                        accumulate_triangle(energy, down_h, start_col, width, height,
                                            strip_height, tri_width, is_half, 0);
                    }
                }

                #pragma omp single
                {
                    current_height = down_h - 1;
                }

                #pragma omp barrier
            }
        }

        if (current_height >= 0) {
            for (int i = current_height; i >= 0; i--) {
                for (int j = 0; j < width; j++) {
                    accumulate_single(energy, i, j, width);
                }
            }
        }
    }
}

void find_seam(const float *energy, int width, int height, int *seam) {
    int min_j = 0;

    for (int j = 1; j < width; j++) {
        if (energy[j] < energy[min_j]) {
            min_j = j;
        }
    }
    seam[0] = min_j;

    for (int i = 1; i < height; i++) {
        int prev_j = seam[i - 1];
        int best_j = prev_j;
        float best_val = energy[i * width + prev_j];

        if (prev_j > 0) {
            float left_val = energy[i * width + (prev_j - 1)];
            if (left_val < best_val) {
                best_val = left_val;
                best_j = prev_j - 1;
            }
        }

        if (prev_j < width - 1) {
            float right_val = energy[i * width + (prev_j + 1)];
            if (right_val < best_val) {
                best_val = right_val;
                best_j = prev_j + 1;
            }
        }

        seam[i] = best_j;
    }
}

unsigned char *remove_seam(const unsigned char *image, int width, int height, int cpp, const int *seam, int use_parallel) {
    size_t out_size = (size_t)(width - 1) * height * cpp * sizeof(unsigned char);
    unsigned char *out = malloc(out_size);
    if (out == NULL) {
        return NULL;
    }

    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int i = 0; i < height; i++) {
        int out_j = 0;
        for (int j = 0; j < width; j++) {
            if (j == seam[i]) {
                continue;
            }

            for (int ch = 0; ch < cpp; ch++) {
                out[(i * (width - 1) + out_j) * cpp + ch] =
                    image[(i * width + j) * cpp + ch];
            }
            out_j++;
        }
    }

    return out;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("USAGE: %s input_image output_image [num_pixels] [use_parallel]\n", argv[0]);
        return EXIT_FAILURE;
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    int n_pixels = 128;
    if (argc >= 4) {
        char *endptr = NULL;
        long parsed = strtol(argv[3], &endptr, 10);
        if (*argv[3] == '\0' || *endptr != '\0' || parsed <= 0 || parsed > INT_MAX) {
            printf("Invalid number of pixels to remove: %s\n", argv[3]);
            return EXIT_FAILURE;
        }
        n_pixels = (int)parsed;
    }

    int use_parallel = 1;
    if (argc >= 5) {
        char *endptr = NULL;
        long parsed = strtol(argv[4], &endptr, 10);
        if (*argv[4] == '\0' || *endptr != '\0' || (parsed != 0 && parsed != 1)) {
            printf("Invalid use_parallel flag: %s (expected 0 or 1)\n", argv[4]);
            return EXIT_FAILURE;
        }
        use_parallel = (int)parsed;
    }

    int use_triangles = 1;
    if (argc >= 6) {
        char *endptr = NULL;
        long parsed = strtol(argv[5], &endptr, 10);
        if (*argv[5] == '\0' || *endptr != '\0' || (parsed != 0 && parsed != 1)) {
            printf("Invalid use_triangles flag: %s (expected 0 or 1)\n", argv[5]);
            return EXIT_FAILURE;
        }
        use_triangles = (int)parsed;
    }

    int triangle_width = 64;
    if (argc >= 7) {
        char *endptr = NULL;
        long parsed = strtol(argv[6], &endptr, 10);

        if (*argv[6] == '\0' || *endptr != '\0' || parsed <= 0) {
            printf("Invalid triangle_width: %s (expected a positive integer)\n", argv[6]);
            return EXIT_FAILURE;
        }

        triangle_width = (int)parsed;
    }


    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    if (image_in == NULL) {
        printf("Failed to load image: %s\n", image_in_name);
        return EXIT_FAILURE;
    }

    if (n_pixels >= width) {
        printf("Cannot remove %d pixels from width %d\n", n_pixels, width);
        stbi_image_free(image_in);
        return EXIT_FAILURE;
    }

    printf("Width: %d, Height: %d, Channels: %d\n", width, height, cpp);
    printf("Removing %d pixels from image %s, saving into image %s\n", n_pixels, image_in_name, image_out_name);
    printf("Parallel: %d\n", use_parallel);
    printf("Use triangles: %d\n", use_triangles);

    printf("Triangle width %d\n", triangle_width);

    unsigned char *current_image = image_in;
    int current_width = width;
    int current_is_malloc = 0;

    double start = omp_get_wtime();
    for (int s = 0; s < n_pixels; s++) {
        float *energy = malloc((size_t)current_width * height * sizeof(float));
        int *seam = malloc((size_t)height * sizeof(int));

        if (energy == NULL || seam == NULL) {
            printf("Error: Failed to allocate energy/seam buffers\n");
            free(energy);
            free(seam);
            if (current_is_malloc) free(current_image);
            else stbi_image_free(current_image);
            return EXIT_FAILURE;
        }

        compute_energy(current_image, current_width, height, cpp, energy, use_parallel);
        accumulate_energy(energy, current_width, height, use_parallel, use_triangles, triangle_width);
        find_seam(energy, current_width, height, seam);

        unsigned char *next_image = remove_seam(current_image, current_width, height, cpp, seam, use_parallel);

        free(energy);
        free(seam);

        if (next_image == NULL) {
            printf("Error: Failed to allocate output image\n");
            if (current_is_malloc) free(current_image);
            else stbi_image_free(current_image);
            return EXIT_FAILURE;
        }

        if (current_is_malloc) free(current_image);
        else stbi_image_free(current_image);

        current_image = next_image;
        current_is_malloc = 1;
        current_width--;
    }
    double stop = omp_get_wtime();
    printf("runtime_seconds=%.6f\n", stop - start);

    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found\n");
        free(current_image);
        return EXIT_FAILURE;
    }
    file_type++;

    int ok = 0;
    if (!strcmp(file_type, "png")) {
        ok = stbi_write_png(image_out_name, current_width, height, cpp, current_image, current_width * cpp);
    } else if (!strcmp(file_type, "jpg") || !strcmp(file_type, "jpeg")) {
        ok = stbi_write_jpg(image_out_name, current_width, height, cpp, current_image, 100);
    } else if (!strcmp(file_type, "bmp")) {
        ok = stbi_write_bmp(image_out_name, current_width, height, cpp, current_image);
    } else {
        printf("Error: Unknown image format %s! Only png, jpg/jpeg, or bmp supported.\n", file_type);
        free(current_image);
        return EXIT_FAILURE;
    }

    if (!ok) {
        printf("Failed to write output image: %s\n", image_out_name);
        free(current_image);
        return EXIT_FAILURE;
    }

    free(current_image);
    return EXIT_SUCCESS;
}