#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiplyMatrices(int **A, int **B, int **C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0;
            for (int k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int **allocateMatrix(int rows, int cols) {
    int **mat = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        mat[i] = (int *)malloc(cols * sizeof(int));
    }
    return mat;
}

void freeMatrix(int **mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

void DO(int n,int m,int p){
    int **A = allocateMatrix(n, m);
    int **B = allocateMatrix(m, p);
    int **C = allocateMatrix(n, p);

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i][j] = rand() % 10;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            B[i][j] = rand() % 10;
        }
    }

    clock_t start = clock();
    multiplyMatrices(A, B, C, n, m, p);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Matrix multiplication (%dx%d) * (%dx%d) completed.\n", n, m, m, p);
    printf("Execution time: %f seconds\n", time_taken);

    freeMatrix(A, n);
    freeMatrix(B, m);
    freeMatrix(C, n);
}
int main(int argc, char* argv[]) {

    DO(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
    
    return 0;
}
