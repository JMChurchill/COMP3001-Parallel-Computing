// Question2Imp1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
/* 
To run:
go into folder:
cd .\Question2_Imp1_2\
compile using:
cl -O2 Question2_Imp1_2.cpp -openmp:experimental
run using
.\Question2_Imp1_2.exe
*/
#include <Windows.h>
#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>

#include <omp.h>

#define BILLION 1000000000
#define TIMES 50000

#define N 128

//#define ARITHMETICAL_OPS 5*N*N*N
//#define ARITHMETICAL_OPS 344100945960//4098
//#define ARITHMETICAL_OPS 42949672960//2048
//#define ARITHMETICAL_OPS 5368709120//1024
//#define ARITHMETICAL_OPS 671088640//512
//#define ARITHMETICAL_OPS 83886080//256
#define ARITHMETICAL_OPS 10485760//128
//#define ARITHMETICAL_OPS 1310720//64



void imp_1();
void imp_2();
void MMM_init();

__declspec(align(64)) float C[N * N], test[N * N], A[N * N], B[N * N]; //square matrixes are considered only, stored as 1d arrays


//__declspec(align(64)) float  X[N], Y[N];
//__declspec(align(64)) float A[N][N], Atr[N * N];


int main()
{
    //std::cout << "Hello World!\n";
    double my_flops;

    MMM_init();

    //the following command pins the current process to the 1st core
     //otherwise, the OS tongles this process between different cores
    //BOOL success = SetProcessAffinityMask(GetCurrentProcess(), 1);
    //if (success == 0) {
    //    //cout << "SetProcessAffinityMask failed" << endl;
    //    printf("\nSetProcessAffinityMask failed\n");
    //    system("pause");
    //    return -1;
    //}

    //define the timers measuring execution time
    clock_t start_1, end_1; //ignore this for  now
    start_1 = clock();

    for (int t = 0; t < TIMES; t++)
    {
        //imp_1();
        imp_2();
    }
    end_1 = clock(); //end the timer
    printf("elapsed time = %f seconds\n", (float)(end_1 - start_1) / CLOCKS_PER_SEC);
    printf("\n %d", N);
    printf("\n start: %f", (float)start_1);
    printf("\n end: %f", (float)end_1);
    printf("\n TIMES: %f", (float)TIMES);
    printf("\n OPS: %d", ARITHMETICAL_OPS);


    //printf("\n The first and last values are %f %f\n", Y[0], Y[N - 1]);


    my_flops = (double)(TIMES * (double)((ARITHMETICAL_OPS) / ((end_1) / CLOCKS_PER_SEC))); 
    printf("\n%f GigaFLOPS achieved\n", my_flops / BILLION);

    return 0;
}

void imp_1() {
    //implementation #1
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[N * i + j] += A[N * i + k] * B[N * k + j];
}

void imp_2() {
    //implementation #2
    //printf("\nimplementation 2");

    //__declspec(align(64)) float C[N * N], A[N * N], B[N * N];

    float tmp;
    int i, j, k;

#pragma omp parallel  

    {

#pragma omp for private(i, j, k, tmp)

        for (i = 0; i < N; i++) {

            for (j = 0; j < N; j++) {

                tmp = 0.0;

#pragma omp simd reduction(+:tmp) aligned(C,A,B:64)

                for (k = 0; k < N; k++) {

                    tmp += A[N * i + k] * B[N * k + j];

                }

                C[N * i + j] = tmp;

            }

        }

    }
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file



void MMM_init() {

    float e = 0.1234, p = 0.7264, r = 0.11;

    //MMM
    for (unsigned int i = 0; i < N; i++) { //printf("\n");
        for (unsigned int j = 0; j < N; j++) {
            C[N * i + j] = 0.0;
            test[N * i + j] = 0.0;
            A[N * i + j] = (j % 9) + p; //printf(" %3.1f",A[i][j]);
            B[N * i + j] = (j % 7) - p; //printf(" %3.1f",B[i][j]);
        }
    }


}