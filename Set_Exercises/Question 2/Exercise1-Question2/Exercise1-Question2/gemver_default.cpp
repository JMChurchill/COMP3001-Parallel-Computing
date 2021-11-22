/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <stdio.h> //this library is needed for printf function
#include <stdlib.h> //this library is needed for rand() function
#include <windows.h> //this library is needed for pause() function
#include <time.h> //this library is needed for clock() function
#include <math.h> //this library is needed for abs()
#include <pmmintrin.h>
#include <process.h>
//#include <chrono>
#include <iostream>
#include <immintrin.h>

void initialize();
void initialize_again();
void slow_routine(float alpha, float beta);//you will optimize this routine
void original_routine(float alpha, float beta);
unsigned short int Compare(float alpha, float beta);
unsigned short int equal(float const a, float const b);

#define N 8192 //input size -----> all input sizes work in debug mode however when ran in release mode an error occurs 
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 1 //how many times the function will run
#define EPSILON 0.0001

#define BILLION 1000000000
#define ARITHMETICAL_OPS ((N*N*4)+(N*N*3)+(N*N*3)+(N)) 
//#define T (N)



// Optimisations applied

//loop merge



int main() {

	float alpha = 0.23f, beta = 0.45f;
	double my_flops;

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now

	initialize();

	start_1 = clock(); //start the timer 

	for (int i = 0; i < TIMES_TO_RUN; i++) {//this loop is needed to get an accurate ex.time value
		slow_routine(alpha, beta);//improved routine -> GigaFLOPS 3.793 -> N = 8192, TIMES_TO_RUN = 130 -> took 20.1 seconds
		//original_routine(alpha, beta);//original -> GigaFLOPS 1.167 -> N = 8192, TIMES_TO_RUN = 40 -> took 23.2 seconds
	}

	end_1 = clock(); //end the timer 

	printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time

	//my_flops = (double)(TIMES_TO_RUN * (double)((ARITHMETICAL_OPS) / ((end_1) / CLOCKS_PER_SEC)));
	//printf("%f", (double)ARITHMETICAL_OPS);
	//printf("\n%f GigaFLOPS achieved\n", my_flops / BILLION);
	//printf("%d\n", my_flops);

	if (Compare(alpha, beta) == 0){
		printf("\nCorrect Result\n");
	}
	else
		printf("\nINcorrect Result\n");

	system("pause"); //this command does not let the output window to close

	return 0; //normally, by returning zero, we mean that the program ended successfully. 
}


void initialize() {

	unsigned int    i, j;

	//initialization
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			A[i][j] = 1.1f;

		}

	for (i = 0; i < N; i++) {
		z[i] = (i % 9) * 0.8f;
		x[i] = 0.1f;
		u1[i] = (i % 9) * 0.2f;
		u2[i] = (i % 9) * 0.3f;
		v1[i] = (i % 9) * 0.4f;
		v2[i] = (i % 9) * 0.5f;
		w[i] = 0.0f;
		y[i] = (i % 9) * 0.7f;
	}

}

void initialize_again() {

	unsigned int    i, j;

	//initialization
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			A[i][j] = 1.1f;

		}

	for (i = 0; i < N; i++) {
		z[i] = (i % 9) * 0.8f;
		x[i] = 0.1f;
		test[i] = 0.0f;
		u1[i] = (i % 9) * 0.2f;
		u2[i] = (i % 9) * 0.3f;
		v1[i] = (i % 9) * 0.4f;
		v2[i] = (i % 9) * 0.5f;
		y[i] = (i % 9) * 0.7f;
	}

}

//you will optimize this routine
void slow_routine(float alpha, float beta) {
	int ii, jj;
	int T = N;
	int upperBound = (N / 4) * 4;
	int upperBoundTile = ((N / T) * T);
	unsigned int i, j;
	for (ii = 0; ii < upperBoundTile; ii += T) {
		for (jj = 0; jj < upperBoundTile; jj += T) {
			__m128 num0, num1, num2, num3, num4;
			__m128 num7, num8, num9, num10, num11, temp;
			temp = _mm_set_ps(0.45f, 0.45f, 0.45f, 0.45f);
			//for (i = 0; i < N; i++) {
			for (i = ii; i < ii + T; i++) {
				num1 = _mm_load_ps1(&u1[i]);
				num3 = _mm_load_ps1(&u2[i]);
				num7 = _mm_load_ps1(&y[i]);

				//for (j = 0; j < (((N) / 4) * 4); j += 4) {
				for (j = jj; j < (((jj + T) / 4) * 4); j += 4) {
					num4 = _mm_load_ps(&v2[j]);
					num0 = _mm_load_ps(&A[i][j]);
					num2 = _mm_load_ps(&v1[j]);

					num0 = _mm_fmadd_ps(num1, num2, num0);//u1[i]*v1[j] + A[i][j]
					num0 = _mm_fmadd_ps(num3, num4, num0);//u2[i]*v2[j]+(u1[i]*v1[j]+A[i][j])
					_mm_store_ps(&A[i][j], num0);
					//A[i][j] = A[i][j] + (u1[i] * v1[j]) + (u2[i] * v2[j]);

					num8 = _mm_load_ps(&A[i][j]);
					num9 = _mm_load_ps(&x[j]);

					num10 = _mm_mul_ps(num8, num7);//(A[j][i]*y[j])

					num11 = _mm_fmadd_ps(num10, temp, num9); //(A[j][i] * y[j])*(0.45) + x[i]

					_mm_store_ps((float*)&x[j], num11);
				}
				//for (; j < N; j++)
				for (; j < jj + T; j++)
				{
					A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
					//printf("not 4"); 
					x[j] = x[j] + 0.45f * A[i][j] * y[i];
				}
			}
		}
		for (j; j < N; j++) {
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
			x[j] = x[j] + beta * A[i][j] * y[i];
		}

	}
	for (; ii < N; ii++) {
		//for (; jj < N; jj++) {
			for (i = ii; i < N; i++)
				for (j = 0; j < N; j++){
					A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
					x[j] = x[j] + beta * A[i][j] * y[i];
				}
		//}
	}
		for (ii = 0; ii < upperBoundTile; ii += T) {
			__m128 num12, num13, num14;
			//for (i = 0; i < (((N) / 4) * 4); i += 4) {
				for (i = ii; i < (((ii + T) / 4) * 4); i += 4) {
				num12 = _mm_load_ps(&x[i]);
				num13 = _mm_load_ps(&z[i]);
				num14 = _mm_add_ps(num12, num13);//(x[i]+z[i])
				_mm_store_ps(&x[i], num14);
			}
			//for (; i < N; i++) {
				for (; i < ii + T; i++){
				x[i] = x[i] + z[i];
			}
		}
		for (; ii < N; ii++) {
			for (i = ii; i < N; i++) {
				x[i] = x[i] + z[i];
			}
		}
		//for (ii = 0; ii < upperBoundTile; ii += T) {
		//	for (jj = 0; jj < upperBoundTile; jj += T) {
				__m128 num15, num16, num17, num18, num19, temp3;
				temp3 = _mm_set_ps(0.23f, 0.23f, 0.23f, 0.23f);
				for (i = 0; i < N; i++) {
					//for (i = ii; i < ii + T; i++) {
					num15 = _mm_load_ps1(&w[i]);
					for (j = 0; j < (((N) / 4) * 4); j += 4) {
						//for (j = jj; j < (((jj + T) / 4) * 4); j += 4) {
						num16 = _mm_load_ps(&A[i][j]);
						num17 = _mm_load_ps(&x[j]);

						num18 = _mm_mul_ps(num16, num17);//(A[i][j] * x[j])

						num15 = _mm_fmadd_ps(temp3, num18, num15);//(0.23*(A[i][j] * x[j]))+w[i]
						//w[i] = w[i] + (0.23f * A[i][j] * x[j]);//wrote alph as literal
					}
					num19 = _mm_hadd_ps(num15, num15);
					num19 = _mm_hadd_ps(num19, num19);

					_mm_store_ss((float*)&w[i], num19);
					for (; j < N; j++) {
						//for (; j < jj + T; j++) {
						w[i] = w[i] + 0.23f * A[i][j] * x[j];
					}
				}
			//}
		//	for (j; j < N; j++) {
		//		A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		//		x[j] = x[j] + beta * A[i][j] * y[i];
		//	}
		//}
		//for (; ii < N; ii++) {
		//	for (jj = 0; jj < N; jj++) {
		//		for (i = ii; i < N; i++) {
		//			for (j = 0; j < N; j++) {
		//				//x[i] = x[i] + z[i];
		//				w[i] = w[i] + alpha * A[i][j] * x[j];
		//			}
		//		}
		//	}
		//}
}

void original_routine(float alpha, float beta) {

	unsigned int i, j;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
			//x[j] = x[j] + beta * A[i][j] * y[i];
		}



	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			x[i] = x[i] + beta * A[j][i] * y[j];

	for (i = 0; i < N; i++)
		x[i] = x[i] + z[i];


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			w[i] = w[i] + alpha * A[i][j] * x[j];


}


unsigned short int Compare(float alpha, float beta) {

	unsigned int i, j;

	initialize_again();


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			x[i] = x[i] + beta * A[j][i] * y[j];

	for (i = 0; i < N; i++)
		x[i] = x[i] + z[i];


	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			test[i] = test[i] + alpha * A[i][j] * x[j];
		}
	}



	for (j = 0; j < N; j++) {
		if (equal(w[j], test[j]) == 1) {
			printf("\n %f %f", test[j], w[j]);
			return -1;
		}
	}

	return 0;
}




unsigned short int equal(float const a, float const b) {

	if (fabs(a - b) / fabs(a) < EPSILON)
		return 0; //success
	else
		return 1;
}



