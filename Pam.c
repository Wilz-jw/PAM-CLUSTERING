#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <stddef.h>
#include <ctype.h> // Added for isdigit function

#define MAX_SPECIES_NAME_LENGTH 20
#define NUM_FEATURES 4
#define MAX_LINE_LENGTH 100

typedef struct {
    double features[NUM_FEATURES];
    char species[MAX_SPECIES_NAME_LENGTH];
} DataPoint;

typedef struct {
    DataPoint *points;
    int numPoints;
} Dataset;

typedef struct {
    int *medoidIndices; // Indices of the medoid points
    int *clusterAssignment; // Cluster assignment for each point
    int numClusters;
} Clusters;

void readDataFromFile(const char *filename, Dataset *dataset) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;

    // Count the number of lines (data points)
    while (fgets(line, sizeof(line), file)) {
        count++;
    }
    rewind(file); // Reset file pointer to the beginning

    // Allocate memory for data points
    dataset->points = (DataPoint *)malloc(count * sizeof(DataPoint));
    dataset->numPoints = count;

    // Read the data points
    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        sscanf(line, "%lf,%lf,%lf,%lf,%[^,\n]",
               &dataset->points[i].features[0],
               &dataset->points[i].features[1],
               &dataset->points[i].features[2],
               &dataset->points[i].features[3],
               dataset->points[i].species);
        i++;
    }

    fclose(file);
}

// Minkowski distance
double minkowskiDistance(double *a, double *b, int length, double p) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: Null pointer detected in minkowskiDistance function.\n");
        exit(EXIT_FAILURE); // Exit or handle the error appropriately
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += pow(fabs(a[i] - b[i]), p);
    }
    return pow(sum, 1.0 / p);
}

// Pearson correlation
double pearsonCorrelation(double *a, double *b, int length) {
    double sum_a = 0.0, sum_b = 0.0, sum_a_sq = 0.0, sum_b_sq = 0.0, sum_ab = 0.0;
    for (int i = 0; i < length; i++) {
        sum_a += a[i];
        sum_b += b[i];
        sum_a_sq += a[i] * a[i];
        sum_b_sq += b[i] * b[i];
        sum_ab += a[i] * b[i];
    }
    double numerator = sum_ab - (sum_a * sum_b / length);
    double denominator = sqrt((sum_a_sq - sum_a * sum_a / length) * (sum_b_sq - sum_b * sum_b / length));
    return denominator == 0 ? 0 : numerator / denominator;
}

// Similarity measure using Pearson correlation (convert to distance)
double pearsonDistance(double *a, double *b, int length) {
    return 1.0 - pearsonCorrelation(a, b, length);
}

// Wrapper for Pearson distance to match the function signature
double pearsonDistanceWrapper(double *a, double *b, int length, double p) {
    return pearsonDistance(a, b, length);
}

// Initialize medoids randomly
void initializeMedoids(int *medoids, int numMedoids, int numPoints) {
    for (int i = 0; i < numMedoids; i++) {
        int randomIndex;
        int isUnique;
        do {
            isUnique = 1;
            randomIndex = rand() % numPoints;
            for (int j = 0; j < i; j++) {
                if (medoids[j] == randomIndex) {
                    isUnique = 0;
                    break;
                }
            }
        } while (!isUnique);
        medoids[i] = randomIndex;
    }
}

// K-Medoids Clustering
void kMedoids(Dataset *dataset, Clusters *clusters, int k, double (*distanceFunc)(double*, double*, int, double), double p) {
    printf("\n\nPAM Clustering Results\n");
    printf("=======================\n\n");

    int n = dataset->numPoints;
    int *medoids = clusters->medoidIndices;
    int *assignment = clusters->clusterAssignment;

    initializeMedoids(medoids, k, n);

    int changed;
    double bestCost = INFINITY;
    do {
        changed = 0;
        double totalCost = 0.0;

        // Assignment step: Assign each point to the nearest medoid
        for (int i = 0; i < n; i++) {
            double minDist = INFINITY;
            for (int j = 0; j < k; j++) {
                double dist = distanceFunc(dataset->points[i].features, dataset->points[medoids[j]].features, NUM_FEATURES, p);
                if (dist < minDist) {
                    minDist = dist;
                    assignment[i] = j;
                }
            }
            totalCost += minDist;
        }

        if (totalCost < bestCost) {
            bestCost = totalCost;
        }
     // Update step: Choose the best medoid for each cluster
        for (int j = 0; j < k; j++) {
            double minTotalDist = INFINITY;
            int bestMedoid = medoids[j];
            for (int i = 0; i < n; i++) {
                if (assignment[i] == j) {
                    double totalDist = 0.0;
                    for (int m = 0; m < n; m++) {
                        if (assignment[m] == j) {
                            totalDist += distanceFunc(dataset->points[i].features, dataset->points[m].features, NUM_FEATURES, p);
                        }
                    }
                    if (totalDist < minTotalDist) {
                        minTotalDist = totalDist;
                        bestMedoid = i;
                    }
                }
            }
            if (medoids[j] != bestMedoid) {
                medoids[j] = bestMedoid;
                changed = 1;
            }
        }
    } while (changed);

    clusters->medoidIndices = medoids;
    clusters->clusterAssignment = assignment;
    clusters->numClusters = k;

    // Print to command line
    printf("Cluster Assignments:\n");
    printf("--------------------\n\n");
    for (int i = 0; i < n; i++) {
        printf("Data Point %d: Cluster %d\n", i, assignment[i]);
    }
    printf("\n");

    // Final Medoids
    printf("Final Central Medoids:\n");
    printf("======================n\n");
    for (int i = 0; i < k; i++) {
        printf("Cluster %d:Data Point (", i );
        for (int j = 0; j < NUM_FEATURES; j++) {
            printf("%.2f", dataset->points[medoids[i]].features[j]);
            if (j < NUM_FEATURES - 1) {
                printf(", ");
            }
        }
        printf(")\n\n");
    }
    printf("Best Cost: %.3f\n", bestCost);

    printf("\n\n");
}

void printClusters(Dataset *dataset, Clusters *clusters) {
     printf("========================================\n");
        printf("CLUSTER Grouping With Final Assignments:\n");
        printf("========================================\n\n");
    for (int i = 0; i < clusters->numClusters; i++) {   
        printf("Cluster %d (Medoid: Point %d):\n", i, clusters->medoidIndices[i]);
        for (int j = 0; j < dataset->numPoints; j++) {
            if (clusters->clusterAssignment[j] == i) {
                printf("  Point %d: (%.2lf, %.2lf, %.2lf, %.2lf) - %s\n",
                       j,
                       dataset->points[j].features[0],
                       dataset->points[j].features[1],
                       dataset->points[j].features[2],
                       dataset->points[j].features[3],
                       dataset->points[j].species);
            }
        }
        printf("\n");
        
    }
}

int main() {
    Dataset dataset;
    readDataFromFile("iris.data", &dataset);

    int k;
    while (1) {
        char input[100];
        printf("Enter the number of clusters (k): ");
        if (fgets(input, sizeof(input), stdin) != NULL) {
            char *endptr;
            long int value = strtol(input, &endptr, 10);
            if (endptr == input || !isspace(*endptr)) {
                printf("Invalid input. Please enter a positive integer.\n\n");
                continue;
            }
            if (value <= 0 || value > INT_MAX) {
                printf("Invalid input. Please enter a positive integer within the range.\n\n");
                continue;
            }
            k = (int)value;
            break;
        } else {
            printf("Error reading input.\n\n");
            return 1;
        }
    }

    int metric;
    while (1) {
        printf("Choose the distance metric:\n");
        printf("1. Minkowski distance\n");
        printf("2. Pearson distance\n");
        printf("Enter your choice (1 or 2): ");
        char input[100];
        if (fgets(input, sizeof(input), stdin) != NULL) {
            char *endptr;
            long int value = strtol(input, &endptr, 10);
            if (endptr == input || !isspace(*endptr)) {
                printf("Invalid input. Please enter 1 or 2.\n\n");
                continue;
            }
            if (value < 1 || value > 2) {
                printf("Invalid input. Please enter 1 or 2.\n\n");
                continue;
            }
            metric = (int)value;
            break;
        } else {
            printf("Error reading input.\n\n");
            return 1;
        }
    }

    double (*distanceFunc)(double*, double*, int, double) = NULL;
    double p = 2.0; // Default value for Minkowski distance

    if (metric == 1) {
        printf("\n\n==============================================================\n");
        printf("THIS IS THE RESULT FOR PAM CLUSTERING WITH MINKOWSKI DISTANCE:\n");
        printf("==============================================================\n\n");
        distanceFunc = minkowskiDistance;
        p = 2.0; // Minkowski distance with p=2 (Euclidean distance)
    } else if (metric == 2) {
        printf("\n\n ============================================================\n");
        printf(" THIS IS THE RESULT FOR PAM CLUSTERING WITH PEARSON DISTANCE:\n ");
        printf("============================================================\n\n");
        distanceFunc = pearsonDistanceWrapper;
    }

    Clusters clusters;
    clusters.numClusters = k;
    clusters.medoidIndices = (int *)malloc(k * sizeof(int));
    clusters.clusterAssignment = (int *)malloc(dataset.numPoints * sizeof(int));

    // Seed random number generator for reproducibility (use srand(time(NULL)) for true randomness)
    srand((unsigned int)time(NULL));

    // Perform K-Medoids clustering
    clock_t start_time = clock();
    kMedoids(&dataset, &clusters, k, distanceFunc, p);
    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print cluster assignments and points in each cluster
    printClusters(&dataset, &clusters);

    printf("Execution Time: %.3f seconds\n\n", execution_time);

    // Free allocated memory
    free(dataset.points);
    free(clusters.medoidIndices);
    free(clusters.clusterAssignment);

    return 0;
}
