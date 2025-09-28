#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void multiplyMatrices(int **A, int **B, int **C, int n, int m, int p) {
    int i, j, k;
    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            C[i][j] = 0;
            for (k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
                
            }
            // printf("Thread %d computed C[%d][%d] = %d\n",
            //    omp_get_thread_num(), i, j, C[i][j]);
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
int DO(int n,int m,int p){

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
    
    double  start = omp_get_wtime();
    multiplyMatrices(A, B, C, n, m, p);
    double end = omp_get_wtime();


    double time_taken = end - start;

    printf("(%dx%d) * (%dx%d) completed, Execution time: %f seconds\n", n, m, m, p, time_taken);

    freeMatrix(A, n);
    freeMatrix(B, m);
    freeMatrix(C, n);

}
int main(int argc, char* argv[]) {

    DO(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
    
    return 0;
}
