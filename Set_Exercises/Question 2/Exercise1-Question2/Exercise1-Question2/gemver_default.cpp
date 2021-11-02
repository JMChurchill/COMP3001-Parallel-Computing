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
unsigned short int Compare(float alpha, float beta);
unsigned short int equal(float const a, float const b);

#define N 8192 //input size
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 1 //how many times the function will run
#define EPSILON 0.0001


// Optimisations applied

//loop merge



int main() {

	float alpha = 0.23f, beta = 0.45f;

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now

	initialize();

	start_1 = clock(); //start the timer 

	for (int i = 0; i < TIMES_TO_RUN; i++)//this loop is needed to get an accurate ex.time value
		slow_routine(alpha, beta);


	end_1 = clock(); //end the timer 

	printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time

	if (Compare(alpha, beta) == 0)
		printf("\nCorrect Result\n");
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

	unsigned int i, j;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}


	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			x[i] = x[i] + 0.45f * A[j][i] * y[j];//wrote beta as literal
		}
		//x[i] = x[i] + z[i];
		//for (j = 0; j < N; j++)
		//	w[i] = w[i] + 0.23f * A[i][j] * x[j];//wrote alph as literal
	}


	//for (i = 0; i < N; i++)
	//	x[i] = x[i] + z[i];

	__m128 num1, num2, num3;
	for (i = 0; i < N; i+=4) {//loop merge bottom two loops
		//x[i] = x[i] + z[i];
		num1 = _mm_load_ps(&x[i]);
		num2 = _mm_load_ps(&z[i]);
		num3 = _mm_add_ps(num1,num2);
		_mm_store_ps(&x[i], num3);

		for (j = 0; j < N; j++){
			w[i] = w[i] + 0.23f * A[i][j] * x[j];//wrote alph as literal
			w[i+1] = w[i+1] + 0.23f * A[i+1][j] * x[j];//wrote alph as literal
			w[i+2] = w[i+2] + 0.23f * A[i+2][j] * x[j];//wrote alph as literal
			w[i+3] = w[i+3] + 0.23f * A[i+3][j] * x[j];//wrote alph as literal
		}
	}
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



