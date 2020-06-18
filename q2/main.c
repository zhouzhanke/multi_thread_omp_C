#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

#include <math.h>
//#include <pthread.h>
#include <omp.h>

#include "monte_carlo.h"
#include "op_timer.h"

#define NUM_POINTS 5000000
#define MAX_THREAD 10

// function 1: f1(x)=cos(x)
// useing built in function call

// function 2: f2(x)=x^2+2x+1
double function_2(double x) {
    double y;
    y = pow(x, 2) + 2 * x + 1;
    
    return y;
}

// function 3: f3(x)=3
double function_3(double x) {
    double y;
    y = 3;
    
    return y;
}

// function 4: f4(x)=10-x
double function_4(double x) {
    double y;
    y = 10 - x;
    
    return y;
}


int main(int argc, char *argv[]) {
    // check for max thread
    if (MAX_THREAD <= 0)
    {
        printf("max thread been set to 0.\n");
        exit(EXIT_SUCCESS);
    }
    
    timer *t = create_timer();
    double result[5][4];
    double timer[MAX_THREAD];
    
    // Explicitly disable dynamic teams
    omp_set_dynamic(0);
    for (int i = 1; i <= MAX_THREAD; i++) {
        omp_set_num_threads(i); // Use i threads for all consecutive parallel regionsX
#pragma omp parallel
        {
//#pragma omp single
//printf("num_threads = %d\n", omp_get_num_threads());
#pragma omp sections
            {
#pragma omp section
                result[0][0] = mc_integrate_1d(cos, NUM_POINTS, 0, 1);
#pragma omp section
                result[0][1] = mc_integrate_1d(function_2, NUM_POINTS, 0, 1);
#pragma omp section
                result[0][2] = mc_integrate_1d(function_3, NUM_POINTS, 0, 1);
#pragma omp section
                result[0][3] = mc_integrate_1d(function_4, NUM_POINTS, 0, 1);
                
#pragma omp section
                result[1][0] = mc_integrate_1d(cos, NUM_POINTS, 0, 2);
#pragma omp section
                result[1][1] = mc_integrate_1d(function_2, NUM_POINTS, 0, 2);
#pragma omp section
                result[1][2] = mc_integrate_1d(function_3, NUM_POINTS, 0, 2);
#pragma omp section
                result[1][3] = mc_integrate_1d(function_4, NUM_POINTS, 0, 2);
                
#pragma omp section
                result[2][0] = mc_integrate_1d(cos, NUM_POINTS, 0, 3);
#pragma omp section
                result[2][1] = mc_integrate_1d(function_2, NUM_POINTS, 0, 3);
#pragma omp section
                result[2][2] = mc_integrate_1d(function_3, NUM_POINTS, 0, 3);
#pragma omp section
                result[2][3] = mc_integrate_1d(function_4, NUM_POINTS, 0, 3);
                
#pragma omp section
                result[3][0] = mc_integrate_1d(cos, NUM_POINTS, 1, 10);
#pragma omp section
                result[3][1] = mc_integrate_1d(function_2, NUM_POINTS, 1, 10);
#pragma omp section
                result[3][2] = mc_integrate_1d(function_3, NUM_POINTS, 1, 10);
#pragma omp section
                result[3][3] = mc_integrate_1d(function_4, NUM_POINTS, 1, 10);
                
#pragma omp section
                result[4][0] = mc_integrate_1d(cos, NUM_POINTS, 0, M_PI);
#pragma omp section
                result[4][1] = mc_integrate_1d(function_2, NUM_POINTS, 0, M_PI);
#pragma omp section
                result[4][2] = mc_integrate_1d(function_3, NUM_POINTS, 0, M_PI);
#pragma omp section
                result[4][3] = mc_integrate_1d(function_4, NUM_POINTS, 0, M_PI);
                
            }
        }
        // get time spend
        timer[i-1] = timer_check(t);
    }
    
    // print result
    // print integral result
    printf("%s,%s,%s,%s,\n", "f1", "f2", "f3", "f4");
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%f,", result[i][j] );
        }
        printf("\n");
    }
    
    // print time record
    printf("%s,%s,\n",  "Threads", "Time");
    for (int i = 0; i < MAX_THREAD; ++i) {
        printf("%d,%f,\n", i+1, timer[i]);
    }
    
    // clean up
    free(t);
    
    exit(EXIT_SUCCESS);
}
