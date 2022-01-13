

#include "canny.h"

unsigned char filt[N][M], gradient[N][M], grad2[N][M], edgeDir[N][M], compEdgeDir[N][M], compGradient[N][M];
unsigned char gaussianMask[5][5];
signed char GxMask[3][3], GyMask[3][3];


void GaussianBlur() {

	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int newPixel;

	unsigned char temp;


	/* Declare Gaussian mask */
	gaussianMask[0][0] = 2;

	gaussianMask[0][1] = 4;
	gaussianMask[0][2] = 5;
	gaussianMask[0][3] = 4;
	gaussianMask[0][4] = 2;

	gaussianMask[1][0] = 4;
	gaussianMask[1][1] = 9;
	gaussianMask[1][2] = 12;
	gaussianMask[1][3] = 9;
	gaussianMask[1][4] = 4;

	gaussianMask[2][0] = 5;
	gaussianMask[2][1] = 12;
	gaussianMask[2][2] = 15;
	gaussianMask[2][3] = 12;
	gaussianMask[2][4] = 5;

	gaussianMask[3][0] = 4;
	gaussianMask[3][1] = 9;
	gaussianMask[3][2] = 12;
	gaussianMask[3][3] = 9;
	gaussianMask[3][4] = 4;

	gaussianMask[4][0] = 2;
	gaussianMask[4][1] = 4;
	gaussianMask[4][2] = 5;
	gaussianMask[4][3] = 4;
	gaussianMask[4][4] = 2;

	/*---------------------- Gaussian Blur ---------------------------------*/
	for (row = 2; row < N - 2; row++) {
		for (col = 2; col < M - 2; col++) {
			newPixel = 0;
			for (rowOffset = -2; rowOffset <= 2; rowOffset++) {
				for (colOffset = -2; colOffset <= 2; colOffset++) {

					newPixel += frame1[row + rowOffset][col + colOffset] * gaussianMask[2 + rowOffset][2 + colOffset];
				}
			}
			filt[row][col] = (unsigned char)(newPixel / 159);
		}
	}


}


void OptimisedSobel() {//optimise this!
	//_mm_loadu_si128(__m128i const* mem_addr) - Load 128-bits of integer data from memory into dst.mem_addr (does not need to be aligned on any particular boundary.)
	//_mm_maddubs_epi16(__m128i a, __m128i b) - vertically multiply each unsigned 8 bit integer from a with the corresponding signed 8-bit from b, producing 
	//											intermediate signed 16-bit integers. Horizontially add adjacent pairs of intermediate signed 16-bit integers, 
	//											and pack the saturated results in dst.
	//_mm_add_epi16(__m128i a, __m128i b) - Add packed 16-bit integers in a and b, and store the results in dst.
	//_mm_hadd_epi16(__m128i a, __m128i b) - Horizontally add adjacent pairs of 16-bit integers in a and b, and pack the signed 16-bit results in dst.
	//_mm_extract_epi16(__m128i a, int imm8) - Extract a 16-bit integer from a, selected with imm8, and store the result in the lower element of dst.
	//_mm_set_epi8(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1,
	//				char e0) - Set packed 8-bit integers in dst with the supplied values.

	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int Gx;
	int Gy;
	float thisAngle;
	int newAngle;
	int newPixel;

	unsigned char temp;


	/* Declare Sobel masks */
	//GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
	//GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
	//GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;

	//GyMask[0][0] = -1; GyMask[0][1] = -2; GyMask[0][2] = -1;
	//GyMask[1][0] = 0; GyMask[1][1] = 0; GyMask[1][2] = 0;
	//GyMask[2][0] = 1; GyMask[2][1] = 2; GyMask[2][2] = 1;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			//for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				//for (colOffset = -1; colOffset <= 1; colOffset++) {

				//	Gx += filt[row + rowOffset][col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
				//	Gy += filt[row + rowOffset][col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				//}
				
				//rowOffset -1
				//colOffset = -1
				Gx += filt[row + -1][col + -1] * -1;
				Gy += filt[row + -1][col + -1] * -1;
				//colOffset = 0
				Gx += filt[row + -1][col + 0] * 0;
				Gy += filt[row + -1][col + 0] * -2;
				//colOffset = 1
				Gx += filt[row + -1][col + 1] * 1;
				Gy += filt[row + -1][col + 1] * -1;



				//rowOffset 0
				//colOffset = -1
				Gx += filt[row + 0][col + -1] * -2;
				Gy += filt[row + 0][col + -1] * 0;
				//colOffset = 0
				Gx += filt[row + 0][col + 0] * 0;
				Gy += filt[row + 0][col + 0] * 0;
				//colOffset = 1
				Gx += filt[row + 0][col + 1] * 2;
				Gy += filt[row + 0][col + 1] * 0;



				//rowOffset 1
				//colOffset = -1
				Gx += filt[row + 1][col + -1] * -1;
				Gy += filt[row + 1][col + -1] * 1;
				//colOffset = 0
				Gx += filt[row + 1][col + 0] * 0;
				Gy += filt[row + 1][col + 0] * 2;
				//colOffset = 1
				Gx += filt[row + 1][col + 1] * 1;
				Gy += filt[row + 1][col + 1] * 1;

			//}

			gradient[row][col] = (unsigned char)(sqrt(Gx * Gx + Gy * Gy));

			thisAngle = (((atan2(Gx, Gy)) / 3.14159) * 180.0);

			/* Convert actual edge direction to approximate value */
			if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
				newAngle = 0;
			else if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
				newAngle = 45;
			else if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
				newAngle = 90;
			else if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
				newAngle = 135;


			edgeDir[row][col] = newAngle;
		}
	}

}

void Sobel() {


	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int Gx;
	int Gy;
	float thisAngle;
	int newAngle;
	int newPixel;

	unsigned char temp;


	/* Declare Sobel masks */
	GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
	GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
	GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;

	GyMask[0][0] = -1; GyMask[0][1] = -2; GyMask[0][2] = -1;
	GyMask[1][0] = 0; GyMask[1][1] = 0; GyMask[1][2] = 0;
	GyMask[2][0] = 1; GyMask[2][1] = 2; GyMask[2][2] = 1;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {

					Gx += filt[row + rowOffset][col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt[row + rowOffset][col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			gradient[row][col] = (unsigned char)(sqrt(Gx * Gx + Gy * Gy));

			thisAngle = (((atan2(Gx, Gy)) / 3.14159) * 180.0);

			/* Convert actual edge direction to approximate value */
			if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
				newAngle = 0;
			else if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
				newAngle = 45;
			else if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
				newAngle = 90;
			else if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
				newAngle = 135;


			edgeDir[row][col] = newAngle;
		}
	}

}


int image_detection() {


	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int Gx;
	int Gy;
	float thisAngle;
	int newAngle;
	int newPixel;

	unsigned char temp;



	/*---------------------- create the image  -----------------------------------*/
	frame1 = (unsigned char**)malloc(N * sizeof(unsigned char*));
	if (frame1 == NULL) { printf("\nerror with malloc fr"); return -1; }
	for (i = 0; i < N; i++) {
		frame1[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
		if (frame1[i] == NULL) { printf("\nerror with malloc fr"); return -1; }
	}


	//create the image
	print = (unsigned char**)malloc(N * sizeof(unsigned char*));
	if (print == NULL) { printf("\nerror with malloc fr"); return -1; }
	for (i = 0; i < N; i++) {
		print[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
		if (print[i] == NULL) { printf("\nerror with malloc fr"); return -1; }
	}

	//initialize the image
	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = 0;

	read_image(IN, frame1);


	GaussianBlur();


	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = filt[i][j];

	write_image(OUT_NAME1, print);

	
	//define the timers measuring execution time
	clock_t start_1, end_1; 
	start_1 = clock();

	//Sobel();
	OptimisedSobel();

	end_1 = clock();
	printf("elapsed time = %f seconds\n", (float)(end_1 - start_1) / CLOCKS_PER_SEC);

	compare();

	/* write gradient to image*/

	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = gradient[i][j];

	write_image(OUT_NAME2, print);



	for (i = 0; i < N; i++)
		free(frame1[i]);
	free(frame1);



	for (i = 0; i < N; i++)
		free(print[i]);
	free(print);


	return 0;

}




void read_image(char filename[], unsigned char** image)
{
	int inint = -1;
	int c;
	FILE* finput;
	int i, j;

	printf("  Reading image from disk (%s)...\n", filename);
	//finput = NULL;
	openfile(filename, &finput);


	for (j = 0; j < N; j++)
		for (i = 0; i < M; i++) {
			c = getc(finput);


			image[j][i] = (unsigned char)c;
		}



	/* for (j=0; j<N; ++j)
	   for (i=0; i<M; ++i) {
		 if (fscanf(finput, "%i", &inint)==EOF) {
		   fprintf(stderr,"Premature EOF\n");
		   exit(-1);
		 } else {
		   image[j][i]= (unsigned char) inint; //printf("\n%d",inint);
		 }
	   }*/

	fclose(finput);

}





void write_image(char* filename, unsigned char** image)
{
	FILE* foutput;
	errno_t err;
	int i, j;


	printf("  Writing result to disk (%s)...\n", filename);
	if ((err = fopen_s(&foutput, filename, "wb")) != NULL) {
		printf("Unable to open file %s for writing\n", filename);
		exit(-1);
	}

	fprintf(foutput, "P2\n");
	fprintf(foutput, "%d %d\n", M, N);
	fprintf(foutput, "%d\n", 255);

	for (j = 0; j < N; ++j) {
		for (i = 0; i < M; ++i) {
			fprintf(foutput, "%3d ", image[j][i]);
			if (i % 32 == 31) fprintf(foutput, "\n");
		}
		if (M % 32 != 0) fprintf(foutput, "\n");
	}
	fclose(foutput);


}










void openfile(char* filename, FILE** finput)
{
	int x0, y0;
	errno_t err;
	char header[255];
	int aa;

	if ((err = fopen_s(finput, filename, "rb")) != NULL) {
		printf("Unable to open file %s for reading\n");
		exit(-1);
	}

	aa = fscanf_s(*finput, "%s", header, 20);

	/*if (strcmp(header,"P2")!=0) {
	   fprintf(stderr,"\nFile %s is not a valid ascii .pgm file (type P2)\n",
			   filename);
	   exit(-1);
	 }*/

	x0 = getint(*finput);
	y0 = getint(*finput);





	if ((x0 != M) || (y0 != N)) {
		printf("Image dimensions do not match: %ix%i expected\n", N, M);
		exit(-1);
	}

	x0 = getint(*finput); /* read and throw away the range info */


}


int getint(FILE* fp) /* adapted from "xv" source code */
{
	int c, i, firstchar, garbage;

	/* note:  if it sees a '#' character, all characters from there to end of
	   line are appended to the comment string */

	   /* skip forward to start of next number */
	c = getc(fp);
	while (1) {
		/* eat comments */
		if (c == '#') {
			/* if we're at a comment, read to end of line */
			char cmt[256], * sp;

			sp = cmt;  firstchar = 1;
			while (1) {
				c = getc(fp);
				if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
				else {
					if (c == '\n' || c == EOF) break;
					if ((sp - cmt) < 250) *sp++ = c;
				}
			}
			*sp++ = '\n';
			*sp = '\0';
		}

		if (c == EOF) return 0;
		if (c >= '0' && c <= '9') break;   /* we've found what we were looking for */

		/* see if we are getting garbage (non-whitespace) */
		if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != ',') garbage = 1;

		c = getc(fp);
	}

	/* we're at the start of a number, continue until we hit a non-number */
	i = 0;
	while (1) {
		i = (i * 10) + (c - '0');
		c = getc(fp);
		if (c == EOF) return i;
		if (c < '0' || c>'9') break;
	}
	return i;
}


int compare() {
	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int Gx;
	int Gy;
	float thisAngle;
	int newAngle;
	int newPixel;

	unsigned char temp;


	/* Declare Sobel masks */
	GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
	GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
	GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;

	GyMask[0][0] = -1; GyMask[0][1] = -2; GyMask[0][2] = -1;
	GyMask[1][0] = 0; GyMask[1][1] = 0; GyMask[1][2] = 0;
	GyMask[2][0] = 1; GyMask[2][1] = 2; GyMask[2][2] = 1;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {

					Gx += filt[row + rowOffset][col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt[row + rowOffset][col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			gradient[row][col] = (unsigned char)(sqrt(Gx * Gx + Gy * Gy));

			thisAngle = (((atan2(Gx, Gy)) / 3.14159) * 180.0);

			/* Convert actual edge direction to approximate value */
			if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
				newAngle = 0;
			else if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
				newAngle = 45;
			else if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
				newAngle = 90;
			else if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
				newAngle = 135;

			compEdgeDir[row][col] = newAngle;
		}
	}

	bool correct = true;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{

			if (compEdgeDir[i][j] == edgeDir[i][j]) {
				//printf("correct");
				//return 1;
			}
			else {
				//printf("wrong");
				//return 0;
				correct = false;
				break;
			}
		}
	}
	//if (compEdgeDir[1][1] == edgeDir[1][1]) {
	//	printf("correct");
	//	return 1;
	//}
	//else {
	//	printf("wrong");
	//	return 0;
	//}
	if (correct)
	{
		printf("\ncorrect\n");
		return 1;
	}
	else {
		printf("\nwrong\n");
		return 0;
	}
}



