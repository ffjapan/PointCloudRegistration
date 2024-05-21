#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <limits>

using namespace Eigen;

#define MAX_LINE_LENGTH 1024

typedef struct {
    double x, y, z, intensity;
} Point;

// Function to read PLY file and load points
int read_ply_file(const char *filename, std::vector<Point> &points) {
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
    //points.resize(num_points);

    // Read points
    for (int i = 0; i < num_points; i++) {
        fscanf(file, "%lf %lf %lf %lf",
               &points[i].x, &points[i].y, &points[i].z, &points[i].intensity);
    }

    fclose(file);
    return num_points;
}

// Function to compute the centroid of a point cloud
Point compute_centroid(const std::vector<Point> &points) {
    Point centroid = {0, 0, 0, 0};
    int num_points = points.size();
    for (int i = 0; i < num_points; i++) {
        centroid.x += points[i].x;
        centroid.y += points[i].y;
        centroid.z += points[i].z;
    }
    centroid.x /= num_points;
    centroid.y /= num_points;
    centroid.z /= num_points;
    return centroid;
}

// Function to subtract the centroid from a point cloud
void subtract_centroid(std::vector<Point> &points, Point centroid) {
    int num_points = points.size();
    for (int i = 0; i < num_points; i++) {
        points[i].x -= centroid.x;
        points[i].y -= centroid.y;
        points[i].z -= centroid.z;
    }
}

// Function to find the nearest neighbor in the target point cloud for each point in the source point cloud
void find_nearest_neighbors(const std::vector<Point> &source, const std::vector<Point> &target, std::vector<int> &indices) {
    int num_points = source.size();
    indices.resize(num_points);
    for (int i = 0; i < num_points; i++) {
        double min_dist = DBL_MAX;
        int min_index = 0;
        for (int j = 0; j < target.size(); j++) {
            double dist = pow(source[i].x - target[j].x, 2) +
                          pow(source[i].y - target[j].y, 2) +
                          pow(source[i].z - target[j].z, 2);
            if (dist < min_dist) {
                min_dist = dist;
                min_index = j;
            }
        }
        indices[i] = min_index;
    }
}

// Function to perform ICP
void icp(std::vector<Point> &source, std::vector<Point> &target, Matrix3d &R, Vector3d &t) {
    Point centroid_source = compute_centroid(source);
    Point centroid_target = compute_centroid(target);

    subtract_centroid(source, centroid_source);
    subtract_centroid(target, centroid_target);

    int max_iterations = 100;
    double tolerance = 1e-6;
    std::vector<int> indices(source.size());

    for (int iter = 0; iter < max_iterations; iter++) {
        find_nearest_neighbors(source, target, indices);

        MatrixXd src(3, source.size());
        MatrixXd tgt(3, source.size());

        for (int i = 0; i < source.size(); i++) {
            src(0, i) = source[i].x;
            src(1, i) = source[i].y;
            src(2, i) = source[i].z;
            tgt(0, i) = target[indices[i]].x;
            tgt(1, i) = target[indices[i]].y;
            tgt(2, i) = target[indices[i]].z;
        }

        MatrixXd H = src * tgt.transpose();
        JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
        Matrix3d U = svd.matrixU();
        Matrix3d V = svd.matrixV();
        Matrix3d R_new = V * U.transpose();

        if (R_new.determinant() < 0) {
            V.col(2) *= -1;
            R_new = V * U.transpose();
        }

        Vector3d t_new = tgt.rowwise().mean() - R_new * src.rowwise().mean();

        double error = (R_new - R).norm() + (t_new - t).norm();
        R = R_new;
        t = t_new;

        if (error < tolerance) break;

        for (int i = 0; i < source.size(); i++) {
            Vector3d p(source[i].x, source[i].y, source[i].z);
            p = R * p + t;
            source[i].x = p(0);
            source[i].y = p(1);
            source[i].z = p(2);
        }

        // Display progress bar
        printf("\rProgress: %d%%", (iter + 1) * 100 / max_iterations);
        fflush(stdout);
    }
    printf("\n");
}

// Example usage
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <source_ply_file> <target_ply_file>\n", argv[0]);
        return -1;
    }

    const char *source_filename = argv[1];
    const char *target_filename = argv[2];
    
    std::vector<Point> source_points, target_points;
    int num_source_points, num_target_points;

    num_source_points = read_ply_file(source_filename, source_points);
    if (num_source_points < 0) {
        return -1;
    }

    num_target_points = read_ply_file(target_filename, target_points);
    if (num_target_points < 0) {
        return -1;
    }

    if (num_source_points != num_target_points) {
        fprintf(stderr, "Error: Source and target point clouds must have the same number of points\n");
        return -1;
    }

    Matrix3d R = Matrix3d::Identity();
    Vector3d t = Vector3d::Zero();

    clock_t start = clock();
    icp(source_points, target_points, R, t);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("ICP completed in %.2f seconds\n", time_taken);

    printf("Rotation matrix:\n");
    std::cout << R << std::endl;

    printf("Translation vector:\n");
    std::cout << t.transpose() << std::endl;

    return 0;
}
