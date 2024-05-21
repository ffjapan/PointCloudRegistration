#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024

typedef struct {
    double x, y, z, intensity;
} Point;

// Function to read PLY file and load points
int read_ply_file(const char *filename, Point **points) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    int num_points = 0;

    // Read header
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "element vertex", 14) == 0) {
            sscanf(line, "element vertex %d", &num_points);
        } else if (strncmp(line, "end_header", 10) == 0) {
            break;
        }
    }

    printf("Number of points: %d\n", num_points);

    // Allocate memory for points
    *points = (Point *)malloc(num_points * sizeof(Point));

    // Temporary variables to read values
    double r, g, b;

    // Read points
    for (int i = 0; i < num_points; i++) {
        fscanf(file, "%lf %lf %lf %lf %lf %lf %lf",
               &(*points)[i].x, &(*points)[i].y, &(*points)[i].z, &r, &g, &b, &(*points)[i].intensity);
    }

    fclose(file);
    return num_points;
}

// Function to write points to a PLY file
void write_ply_file(const char *filename, Point *points, int num_points) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Write PLY header
    fprintf(file, "ply\n");
    fprintf(file, "format ascii 1.0\n");
    fprintf(file, "element vertex %d\n", num_points);
    fprintf(file, "property float x\n");
    fprintf(file, "property float y\n");
    fprintf(file, "property float z\n");
    fprintf(file, "property float intensity\n");
    fprintf(file, "end_header\n");

    // Write point data
    for (int i = 0; i < num_points; i++) {
        fprintf(file, "%f %f %f %f\n",
                points[i].x, points[i].y, points[i].z, points[i].intensity);
    }

    fclose(file);
}

// Function to calculate Euclidean distance between two points
double euclidean_distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}

// Function to perform farthest point sampling
void farthest_point_sampling(Point *points, int num_points, int num_samples, Point *sampled_points) {
    double *distances = (double *)malloc(num_points * sizeof(double));
    for (int i = 0; i < num_points; i++) {
        distances[i] = DBL_MAX;
    }

    srand(time(NULL));
    int initial_index = rand() % num_points;
    sampled_points[0] = points[initial_index];

    for (int i = 1; i < num_samples; i++) {
        // Update distances based on the last sampled point
        for (int j = 0; j < num_points; j++) {
            double dist = euclidean_distance(points[j], sampled_points[i - 1]);
            if (dist < distances[j]) {
                distances[j] = dist;
            }
        }

        // Select the farthest point
        int next_index = -1;
        double max_distance = -1.0;
        for (int j = 0; j < num_points; j++) {
            if (distances[j] > max_distance) {
                max_distance = distances[j];
                next_index = j;
            }
        }
        sampled_points[i] = points[next_index];
        // Display progress bar
        printf("\rProgress: %d%%", (i * 100) / num_samples);
        fflush(stdout);
    }
    printf("\n");

    free(distances);
}

// Example usage
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_ply_file>\n", argv[0]);
        return -1;
    }

    const char *input_filename = argv[1];
    
    // Extract the base filename without extension
    char base_filename[MAX_LINE_LENGTH];
    strncpy(base_filename, input_filename, strlen(input_filename) - 4);
    base_filename[strlen(input_filename) - 4] = '\0';
    
    // Create output filename
    char output_filename[MAX_LINE_LENGTH];
    snprintf(output_filename, sizeof(output_filename), "%s_sampled.ply", base_filename);
    
    Point *points;
    int num_points;

    num_points = read_ply_file(input_filename, &points);
    if (num_points < 0) {
        return -1;
    }

    int num_samples = 10000;
    Point *sampled_points = (Point *)malloc(num_samples * sizeof(Point));

    // Perform FPS and measure time
    clock_t start = clock();
    farthest_point_sampling(points, num_points, num_samples, sampled_points);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("FPS completed in %.2f seconds\n", time_taken);

    // Write sampled points to PLY file
    write_ply_file(output_filename, sampled_points, num_samples);

    // Clean up
    free(points);
    free(sampled_points);

    return 0;
}
