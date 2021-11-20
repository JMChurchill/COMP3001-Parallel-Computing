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

#define N 8192 //input size
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 1 //how many times the function will run
#define EPSILON 0.0001

#define BILLION 1000000000
#define ARITHMETICAL_OPS ((N*N*4)+(N*N*3)+(N*N*3)+(N)) 



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
		slow_routine(alpha, beta);//improved routine
		//original_routine(alpha, beta);
	}

	end_1 = clock(); //end the timer 

	printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time

	if (Compare(alpha, beta) == 0){
		printf("\nCorrect Result\n");
		//my_flops = (double)(TIMES_TO_RUN * (double)((ARITHMETICAL_OPS) / ((end_1) / CLOCKS_PER_SEC)));
		//printf("%f", (double)ARITHMETICAL_OPS);
		//printf("\n%f GigaFLOPS achieved\n", my_flops / BILLION);
		//printf("%d\n", my_flops);
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
	/*
	_mm_load_ps,	// -> load 128bit (composed of 4 packed single-precision (32-bit) floating-point elements) from memory into dst. mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.
	_mm_load_ps1,	// -> Load a single-precision (32-bit) floating-point element from memory into all elements of dst.
	_mm_load_ss,	// -> Load a single-precision (32-bit) floating-point element from memory into the lower of dst, and zero the upper 3 elements. mem_addr does not need to be aligned on any particular boundary.
	_mm_add_ps,		// -> Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
	_mm_mul_ps,		// -> Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
	_mm_store_ps,	// -> Store 128-bits (composed of 4 packed single-precision (32-bit) floating-point elements) from a into memory. mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.
	_mm_store_ss,	// -> Store the lower single-precision (32-bit) floating-point element from a into memory. mem_addr does not need to be aligned on any particular boundary.
	_mm_set1_ps,	// -> Broadcast single-precision (32-bit) floating-point value a to all elements of dst.
	_mm_hadd_ps.	// -> 
	*/
	//int ii, jj;
	//int T = 1;
	//for (ii = 0; ii < N; ii +=T) {
	//	for (jj = 0; jj < N; jj+=T) {
	//		
	//	}
	//}

	unsigned int i, j;

	__m128 num0, num1, num2, num3, num4;
	for (i = 0; i < N; i++) {
		num1 = _mm_load_ps1(&u1[i]);
		num3 = _mm_load_ps1(&u2[i]);

		for (j = 0; j < ((N / 4) * 4); j+=4) {
			num4 = _mm_load_ps(&v2[j]);
			num0 = _mm_load_ps(&A[i][j]);
			num2 = _mm_load_ps(&v1[j]);

			num0 = _mm_fmadd_ps(num1,num2,num0);//u1[i]*v1[j] + A[i][j]
			num0 = _mm_fmadd_ps(num3, num4,num0);//u2[i]*v2[j]+(u1[i]*v1[j]+A[i][j])
			_mm_store_ps(&A[i][j], num0);
			//A[i][j] = A[i][j] + (u1[i] * v1[j]) + (u2[i] * v2[j]);
		}
		for (; j < N; j++) {
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

	__m128 num7, num8, num9, num10, num11, temp;
	temp = _mm_set_ps(0.45f, 0.45f, 0.45f, 0.45f);
	for (j = 0; j < N; j++) {
		num7 = _mm_load_ps1(&y[j]);
		for (i = 0; i < ((N / 4) * 4); i += 4) {
			num8 = _mm_load_ps(&A[j][i]);
			num9 = _mm_load_ps(&x[i]);

			num10 = _mm_mul_ps(num8, num7);//(A[j][i]*y[j])

			num11 = _mm_fmadd_ps(num10, temp, num9); //(A[j][i] * y[j])*(0.45) + x[i]

			_mm_store_ps((float*)&x[i], num11);
		}
		for (; i < N; i++) {
			x[i] = x[i] + beta * A[j][i] * y[j];
		}
	}

	__m128 num12, num13, num14, num15, num16, num17, num18, num19, num20, num21, num22,num23,num24,num25,num26,num27,num28,num29,num30,num31, temp2;
	temp2 = _mm_set_ps(0.23f, 0.23f, 0.23f, 0.23f);
	for (i = 0; i < ((N / 4) * 4); i+=4) {
		num12 = _mm_load_ps(&x[i]);
		num13 = _mm_load_ps(&z[i]);
		num14 = _mm_add_ps(num12, num13);//(x[i]+z[i])
		_mm_store_ps(&x[i], num14);
		num15 = _mm_load_ps1(&w[i]);
		num20 = _mm_load_ps1(&w[i+1]);
		num23 = _mm_load_ps1(&w[i+2]);
		num26 = _mm_load_ps1(&w[i+3]);
		for (j = 0; j < ((N / 4) * 4); j += 4) {
			num17 = _mm_load_ps(&x[j]);

			//calc num15
			num16 = _mm_load_ps(&A[i][j]);

			num18 = _mm_mul_ps(num16, num17);//(A[i][j] * x[j])

			num15 = _mm_fmadd_ps(temp2, num18, num15);//(0.23*(A[i][j] * x[j]))+w[i]
			//calc num20
			num21 = _mm_load_ps(&A[i+1][j]);

			num22 = _mm_mul_ps(num21, num17);//(A[i][j] * x[j])

			num20 = _mm_fmadd_ps(temp2, num22, num20);//(0.23*(A[i][j] * x[j]))+w[i]
			//calc num23
			num24 = _mm_load_ps(&A[i+2][j]);

			num25 = _mm_mul_ps(num24, num17);//(A[i][j] * x[j])

			num23 = _mm_fmadd_ps(temp2, num25, num23);//(0.23*(A[i][j] * x[j]))+w[i]
			//calc num26
			num27 = _mm_load_ps(&A[i+3][j]);

			num28 = _mm_mul_ps(num27, num17);//(A[i][j] * x[j])

			num26 = _mm_fmadd_ps(temp2, num28, num26);//(0.23*(A[i][j] * x[j]))+w[i]
			//if (i == (((N / 4) * 4) - 1)) {
			//	printf("aaa %d\n", j);
			//}
		}
		//if (j < ((N / 4) * 4)) {
		num19 = _mm_hadd_ps(num15, num15);
		num19 = _mm_hadd_ps(num19, num19);

		num29 = _mm_hadd_ps(num20, num20);
		num29 = _mm_hadd_ps(num29, num29);

		num30 = _mm_hadd_ps(num23, num23);
		num30 = _mm_hadd_ps(num30, num30);

		num31 = _mm_hadd_ps(num26, num26);
		num31 = _mm_hadd_ps(num31, num31);

		_mm_store_ss((float*)&w[i], num19);
		_mm_store_ss((float*)&w[i + 1], num29);
		_mm_store_ss((float*)&w[i + 2], num30);
		_mm_store_ss((float*)&w[i + 3], num31);

		for (; j < N; j++) {
			w[i] = w[i] + alpha * A[i][j] * x[j];
			w[i+1] = w[i+1] + alpha * A[i+1][j] * x[j];
			w[i+2] = w[i+2] + alpha * A[i+2][j] * x[j];
			w[i+3] = w[i+3] + alpha * A[i+3][j] * x[j];
		}
	}

	for (i = ((N / 4) * 4); i < N; i++) {
		x[i] = x[i] + z[i];
		//printf("iii %d\n",i);
		for (j = 0; j < N; j++) {
			//printf("jjj %d\n", j);

			w[i] = w[i] + alpha * A[i][j] * x[j];
		}
	}
}

void original_routine(float alpha, float beta) {

	unsigned int i, j;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];


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



