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
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N], Atranspose[N][N];;

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

	//float tempV1, tempV2;
	//for (i = 0; i < N; i++){
	//	tempV1 = u1[i];
	//	tempV2 = u2[i];
	//	for (j = 0; j < N; j++){
	//		A[i][j] = A[i][j] + (tempV1 * v1[j]) + (tempV2 * v2[j]);
	//	}
	//}
	__m128 num0, num1, num2, num3, num4;
	for (i = 0; i < N; i++) {
		num1 = _mm_load_ps1(&u1[i]);
		num3 = _mm_load_ps1(&u2[i]);

		for (j = 0; j < N; j+=4) {
			num4 = _mm_load_ps(&v2[j]);
			num0 = _mm_load_ps(&A[i][j]);
			num2 = _mm_load_ps(&v1[j]);

			num0 = _mm_fmadd_ps(num1,num2,num0);//u1[i]*v1[j] + A[i][j]
			num0 = _mm_fmadd_ps(num3, num4,num0);//u2[i]*v2[j]+(u1[i]*v1[j]+A[i][j])
			_mm_store_ps(&A[i][j], num0);
			//A[i][j] = A[i][j] + (u1[i] * v1[j]) + (u2[i] * v2[j]);
		}
	}

	float tempV3;

	//for (j = 0; j < N; j++)
	//	for (i = 0; i < N; i++)
	//		x[i] = x[i] + beta * A[j][i] * y[j];

	//for (j = 0; j < N; j++) {
	//	tempV3 = y[j];
	//	for (i = 0; i < N; i+=4) {
	//		x[i] = x[i] + 0.45f * A[j][i] * tempV3;
	//		x[i+1] = x[i+1] + 0.45f * A[j][i+1] * tempV3;
	//		x[i+2] = x[i+2] + 0.45f * A[j][i+2] * tempV3;
	//		x[i+3] = x[i+3] + 0.45f * A[j][i+3] * tempV3;
	//	}
	//}

	__m128 num7, num8, num9, num10, num11, temp, temp2;
	temp = _mm_set_ps(0.45f, 0.45f, 0.45f, 0.45f);
	for (j = 0; j < N; j++) {
		num7 = _mm_load_ps1(&y[j]);
		for (i = 0; i < N; i += 4) {
			num8 = _mm_load_ps(&A[j][i]);
			num9 = _mm_load_ps(&x[i]);

			num10 = _mm_mul_ps(num8, num7);//(A[j][i]*y[j])

			num11 = _mm_fmadd_ps(num10, temp, num9); //(A[j][i] * y[j])*(0.45) + x[i]

			_mm_store_ps((float*)&x[i], num11);
		}
	}


	//__m128 num7, num8, num9, num10, num11, temp,temp2;
	//temp = _mm_set_ps(0.45f, 0.45f, 0.45f, 0.45f);
	//for (i = 0; i < N; i++) {//not working-------------------------------------------
	//	num7 = _mm_load_ps1(&x[i]);
	//	for (j = 0; j < N; j+=4) {
	//		//num8 = _mm_load_ps(&A[j][i]);
	//		num8 = _mm_load_ps(&Atranspose[i][j]);
	//		//x[i] = x[i] + (0.45f * A[j][i] * y[j]);//wrote beta as literal
	//		num9 = _mm_load_ps(&y[j]);
	//		num10 = _mm_mul_ps(num8, num9);

	//		num7 = _mm_fmadd_ps(num10, temp, num7);
	//	}
	//	num11 = _mm_hadd_ps(num7, num7);
	//	num11 = _mm_hadd_ps(num11, num11);

	//	_mm_store_ss((float*)&x[i], num11);
	//}

	//for (i = 0; i < N; i++)
	//	x[i] = x[i] + z[i];

	__m128 num12, num13, num14;
	for (i = 0; i < N; i+=4) {
		num12 = _mm_load_ps(&x[i]);
		num13 = _mm_load_ps(&z[i]);
		num14 = _mm_add_ps(num12, num13);//(x[i]+z[i])
		_mm_store_ps(&x[i], num14);
	}
	//float tempV4;
	//for (i = 0; i < N; i++) {//loop merge bottom two loops
	//	tempV4 = w[i];
	//	for (j = 0; j < N; j +=4) {
	//		tempV4 = tempV4 + 0.23f * A[i][j] * x[j];//wrote alph as literal
	//		tempV4 = tempV4 + 0.23f * A[i][j+1] * x[j+1];//wrote alph as literal
	//		tempV4 = tempV4 + 0.23f * A[i][j+2] * x[j+2];//wrote alph as literal
	//		tempV4 = tempV4 + 0.23f * A[i][j+3] * x[j+3];//wrote alph as literal
	//	}
	//	w[i] = tempV4;
	//}
	__m128 num15, num16, num17, num18, num19, temp3;
	temp3 = _mm_set_ps(0.23f, 0.23f, 0.23f, 0.23f);
	for (i = 0; i < N; i++) {
		num15 = _mm_load_ps1(&w[i]);
		for (j = 0; j < N; j += 4) {
			num16 = _mm_load_ps(&A[i][j]);
			num17 = _mm_load_ps(&x[j]);

			num18 = _mm_mul_ps(num16, num17);//(A[i][j] * x[j])

			num15 = _mm_fmadd_ps(temp3, num18, num15);//(0.23*(A[i][j] * x[j]))+w[i]
			//w[i] = w[i] + (0.23f * A[i][j] * x[j]);//wrote alph as literal
		}
		num19 = _mm_hadd_ps(num15, num15);
		num19 = _mm_hadd_ps(num19, num19);

		_mm_store_ss((float*)&w[i], num19);
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



