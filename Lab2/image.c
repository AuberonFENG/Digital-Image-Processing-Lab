#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <ctype.h>
#include "proto.h"

void SavePNMImage(Image*, char*);
Image* SwapImage(Image*);
Image* ReadPNMImage(char*);
Image* CreateNewImage(Image*, char* comment);
Image* ResizeImage(Image* image, float radio);
Image* NegativeImage(Image*);
Image* NearestImage(Image*, int radio1);
Image* AlterReduceImage(Image*, float radio2);
Image* PixelReplicationImage(Image*, float radio3);
Image* BilinearImage(Image*, float radio4);
int TestReadImage(char*, char*);

int main(int argc, char** argv)
{
    TestReadImage("F://DIP//PGM_IMAGES//PGM_IMAGES//lena.pgm", "F://DIP//Lab2//Results//lena_Bilinear.pgm");

    return(0);
}

int TestReadImage(char* filename, char* outfilename)
{
    Image* image;
    Image* outimage;

    image = ReadPNMImage(filename);

    //outimage = ResizeImage(image,3);// for enlargement 1.2, for reduction 0.5
	//outimage = NegativeImage(image);
	//outimage = NearestImage(image, 5);
	//outimage = AlterReduceImage(image, 0.5);
	//outimage = PixelReplicationImage(image, 3);
	outimage = BilinearImage(image, 3);
    SavePNMImage(outimage, outfilename);

    return(0);
}
Image* BilinearImage(Image* image, float radio4){
	unsigned char* tempin, *tempout;
	int i, j, size;
	Image* outimage;
	outimage = (Image*)malloc(sizeof(Image));

	int inputW = image->Width, inputH = image->Height;
	int outputW = inputW * radio4, outputH = inputH * radio4;

	outimage->Width = outputW;
	outimage->Height = outputH;
	outimage->Type = image->Type;

	if (outimage->Type == GRAY)
		size = outimage->Width * outimage->Height;
	else if (outimage->Type == COLOR)
		size = outimage->Width * outimage->Height * 3;

	outimage->data = (unsigned char*)malloc(size);

	tempin = image->data;
	tempout = outimage->data;

	int iLeftTop, jLeftTop, iRightTop, jRightTop, iLeftBottom, jLeftBottom, iRightBottom, jRightBottom;// these variables are original image pixel positions
	unsigned char vLeftTop, vLeftBottom, vRightTop, vRightBottom;
	float p, q;

	for (i = 0; i < outputW; i++) {
		iLeftTop = i / radio4;
		for (j = 0; j < outputH; j++) {
			jLeftTop = j / radio4;
			// get the position of A,B,C,D 
			iRightTop = iLeftTop+1;// after scaling the position is radio4 * iRightTop
			jRightTop = jLeftTop;
			iLeftBottom = iLeftTop;
			jLeftBottom = jLeftTop+1;
			iRightBottom = iLeftTop + 1;
			jRightBottom = jLeftTop + 1;

			// get value of these four pixel
			vLeftTop = *(tempin + iLeftTop* inputW + jLeftTop);
			vLeftBottom = *(tempin + iLeftBottom* inputW + jLeftBottom);
			vRightTop = *(tempin + iRightTop* inputW + jRightTop);
			vRightBottom = *(tempin + iRightBottom*inputW + jRightBottom);

			// get p and q
			p = (j - jLeftTop*radio4) / radio4;
			q = (i - iLeftTop*radio4) / radio4;

			

			// get (i,j) value
			*tempout = (1 - p)*(1 - q)*vLeftTop + p*(1 - q)*vLeftBottom + (1 - p)*q*vRightTop + p*q*vRightBottom;
			/*
			// test
			if (i == 1 && j == 1){
				printf("94,96,91,125\n");
				printf("%d,%d,%d,%d\n", vLeftTop, vRightTop, vLeftBottom, vRightBottom);
				printf("%d,%d,%d,%d\n", *(tempin), *(tempin + 1), *(tempin + inputW), *(tempin + inputW + 1));
				printf("%d,%d\n", p, q);
				printf("%d,%d\n", jLeftTop, iLeftTop);
				printf("%d,%d\n", i, j);

				printf("%d", *tempout);
			}
			*/
			tempout++;
		}
	}

	return(outimage);
}
Image* PixelReplicationImage(Image* image, float radio3){
	unsigned char* tempin, *tempout;
	int i, j, size;
	Image* outimage;
	outimage = (Image*)malloc(sizeof(Image));

	int inputW = image->Width, inputH = image->Height;
	int outputW = inputW * radio3 , outputH = inputH * radio3 ;

	outimage->Width = outputW;
	outimage->Height = outputH;
	outimage->Type = image->Type;

	if (outimage->Type == GRAY)
		size = outimage->Width * outimage->Height;
	else if (outimage->Type == COLOR)
		size = outimage->Width * outimage->Height * 3;

	outimage->data = (unsigned char*)malloc(size);

	tempin = image->data;
	tempout = outimage->data;

	int iBelong, jBelong;

	for (i = 0; i < outputW; i++) {
		iBelong = i / radio3;
		for (j = 0; j < outputH; j++) {
			jBelong = j / radio3;
			*tempout = *(tempin + iBelong * inputW + jBelong);
			tempout++;
		}
	}

	return(outimage);
}
Image* AlterReduceImage(Image* image, float radio2){
	unsigned char* tempin, *tempout;
	int i, j, size;
	
	Image* outimage;
	outimage = (Image*)malloc(sizeof(Image));

	int inputW = image->Width, inputH = image->Height;
	int outputW = (int)(inputW * radio2), outputH = (int)(inputH * radio2);

	outimage->Width = outputW;
	outimage->Height = outputH;
	outimage->Type = image->Type;

	if (outimage->Type == GRAY)
		size = outimage->Width * outimage->Height;
	else if (outimage->Type == COLOR)
		size = outimage->Width * outimage->Height * 3;

	outimage->data = (unsigned char*)malloc(size);

	tempin = image->data;
	tempout = outimage->data;
	int h = 0;
	int w = 0;
	for (i = 1; i <= inputW; i++){
		for (j = 1; j <= inputH; j++){
			w = (int)(i*radio2 - 0.5);
			h = (int)(j*radio2 - 0.5);
			*(tempout + h*outputW + w) = *(tempin + (j-1)*inputW + i-1);

		}
	}

	return(outimage);
}

Image* NearestImage(Image* image, int radio1){
	unsigned char* tempin, *tempout;
	int size;
	Image* outimage;
	outimage = (Image*)malloc(sizeof(Image));

	int inputW = image->Width, inputH = image->Height;
	int outputW = inputW * radio1, outputH = inputH * radio1;

	outimage->Width = outputW;
	outimage->Height = outputH;
	outimage->Type = image->Type;

	if (outimage->Type == GRAY)
		size = outimage->Width * outimage->Height;
	else if (outimage->Type == COLOR)
		size = outimage->Width * outimage->Height * 3;

	outimage->data = (unsigned char*)malloc(size);

	tempin = image->data;
	tempout = outimage->data;
	// THIS IS FOR WITHOUT BOUNDARY
	/*
	for (int i = (radio1-1)/2; i < inputW-(radio1-1)/2; i++){
		for (int j = (radio1 - 1) / 2; j < inputH - (radio1 - 1) / 2; j++){
			for (int k = 0; k < radio1*radio1; k++){
				*(tempout + (i*radio1)*(inputW*radio1) + j*radio1 + 1 - (radio1 - 1) / 2 - (radio1 - 1) / 2 * (inputW*radio1) + k%radio1 + (int)(k / radio1)*(inputW*radio1)) = *(tempin + i*inputW + j);
			}
		}
	}
	*/
	
	for (int i = 1; i <= inputW; i++){
		for (int j = 1; j <= inputH ; j++){
			for (int k = 0; k < radio1*radio1; k++){
				printf("%d,%d\n", j, k);
				*(tempout + (radio1*(j - 1) - (radio1 - 1) / 2)*(inputW*radio1) + (i - 1)*radio1 - (radio1 - 1) / 2 + k%radio1 + (int)(k / radio1)*(inputW*radio1)) = *(tempin + (i - 1)*inputW + (j - 1));
			}
		}
	}
	
	

	return (outimage);


}
Image* NegativeImage(Image* image){
	unsigned char* tempin, *tempout;
	int i, size;
	Image* outimage;

	outimage = CreateNewImage(image, "#testing Nega");
	tempin = image->data;
	tempout = outimage->data;

	if (image->Type == GRAY)   size = image->Width * image->Height;
	else if (image->Type == COLOR) size = image->Width * image->Height * 3;

	
	for(i=0;i<size;i++)
	{
	*tempout=255-(*tempin);

	tempin++;
	tempout++;
	}
	
	return(outimage);
}
Image* SwapImage(Image* image)
{
    unsigned char* tempin, * tempout;
    int i, size;
    Image* outimage;

    outimage = CreateNewImage(image, "#testing Swap");
    tempin = image->data;
    tempout = outimage->data;

    if (image->Type == GRAY)   size = image->Width * image->Height;
    else if (image->Type == COLOR) size = image->Width * image->Height * 3;
	for (i = 0; i<size; i++)
	{
		*tempout = (*tempin);

		tempin++;
		tempout++;
	}
	return(outimage);
}

/*******************************************************************************/
//Read PPM image and return an image pointer                                   
/**************************************************************************/
Image* ReadPNMImage(char* filename)
{
    char ch;
    int  maxval, Width, Height;
    int size, num, j;
    FILE* fp;
    Image* image;
    int num_comment_lines = 0;


    image = (Image*)malloc(sizeof(Image));

    if ((fp = fopen(filename, "rb")) == NULL) {//////r-->rb
        printf("Cannot open %s\n", filename);
        exit(0);
    }

    printf("Loading %s ...", filename);

    if (fscanf(fp, "P%c\n", &ch) != 1) {
        printf("File is not in ppm/pgm raw format; cannot read\n");
        exit(0);
    }
    if (ch != '6' && ch != '5') {
        printf("File is not in ppm/pgm raw format; cannot read\n");
        exit(0);
    }

    if (ch == '5')image->Type = GRAY;  // Gray (pgm)
    else if (ch == '6')image->Type = COLOR;  //Color (ppm)
    /* skip comments */
    ch = getc(fp);
    j = 0;
    while (ch == '#')
    {
        image->comments[num_comment_lines][j] = ch;
        j++;
        do {
            ch = getc(fp);
            image->comments[num_comment_lines][j] = ch;
            j++;
        } while (ch != '\n');     /* read to the end of the line */
        image->comments[num_comment_lines][j - 1] = '\0';
        j = 0;
        num_comment_lines++;
        ch = getc(fp);            /* thanks, Elliot */
    }

    if (!isdigit((int)ch)) {
        printf("Cannot read header information from ppm file");
        exit(0);
    }

    ungetc(ch, fp);               /* put that digit back */

    /* read the width, height, and maximum value for a pixel */
    fscanf(fp, "%d%d%d\n", &Width, &Height, &maxval);

    /*
    if (maxval != 255){
      printf("image is not true-color (24 bit); read failed");
      exit(0);
    }
    */

    if (image->Type == GRAY)
        size = Width * Height;
    else  if (image->Type == COLOR)
        size = Width * Height * 3;
    image->data = (unsigned char*)malloc(size);
    image->Width = Width;
    image->Height = Height;
    image->num_comment_lines = num_comment_lines;

    if (!image->data) {
        printf("cannot allocate memory for new image");
        exit(0);
    }

    num = fread((void*)image->data, 1, (size_t)size, fp);
    printf("\nComplete reading of %d bytes %d size \n", num, size);
    if (num != size) {
        printf("cannot read image data from file");
        exit(0);
    }

    //for(j=0;j<image->num_comment_lines;j++){
    //      printf("%s\n",image->comments[j]);
    //}

    fclose(fp);

    /*-----  Debug  ------*/

    if (image->Type == GRAY)printf("..Image Type PGM\n");
    else printf("..Image Type PPM Color\n");
    /*
    printf("Width %d\n", Width);
    printf("Height %d\n",Height);
    printf("Size of image %d bytes\n",size);
    printf("maxvalue %d\n", maxval);
    */
    return(image);
}

void SavePNMImage(Image* temp_image, char* filename)
{
    int num, j;
    int size;
    FILE* fp;
    //char comment[100];


    printf("Saving Image %s\n", filename);
    fp = fopen(filename, "wb");///w-->wb
    if (!fp) {
        printf("cannot open file for writing");
        exit(0);
    }

    //strcpy(comment,"#Created by Dr Mohamed N. Ahmed");

    if (temp_image->Type == GRAY) {  // Gray (pgm)
        fprintf(fp, "P5\n");
        size = temp_image->Width * temp_image->Height;
    }
    else  if (temp_image->Type == COLOR) {  // Color (ppm)
        fprintf(fp, "P6\n");
        size = temp_image->Width * temp_image->Height * 3;
    }

    for (j = 0; j < temp_image->num_comment_lines; j++)
        fprintf(fp, "%s\n", temp_image->comments[j]);

    fprintf(fp, "%d %d\n%d\n", temp_image->Width, temp_image->Height, 255);

    num = fwrite((void*)temp_image->data, 1, (size_t)size, fp);

    if (num != size) {
        printf("cannot write image data to file");
        exit(0);
    }

    fclose(fp);
}

/*************************************************************************/
/*Create a New Image with same dimensions as input image                 */
/*************************************************************************/

Image* ResizeImage(Image* image, float radio) {// this function can be used to enlargement and reduction
    unsigned char* tempin, * tempout;
    int i, j, size;
    Image* outimage;
    outimage = (Image*)malloc(sizeof(Image));

    int inputW = image->Width, inputH = image->Height;
    int outputW = inputW * radio + 0.5, outputH = inputH * radio + 0.5;

    outimage->Width = outputW;
    outimage->Height = outputH;
    outimage->Type = image->Type;

    if (outimage->Type == GRAY)
        size = outimage->Width * outimage->Height;
    else if (outimage->Type == COLOR)
        size = outimage->Width * outimage->Height * 3;

    outimage->data = (unsigned char*)malloc(size);

    tempin = image->data;
    tempout = outimage->data;

    int iNearest, jNearest;

    for (i = 0; i < outputW; i++) {
        //ioffset = i * outputW;
        iNearest = i / radio + 0.5;
        for (j = 0; j < outputH; j++) {
            jNearest = j / radio + 0.5;
            *tempout = *(tempin + iNearest * inputW + jNearest);
            tempout++;
        }
    }

    return(outimage);
}

Image* CreateNewImage(Image* image, char* comment)
{
    Image* outimage;
    int size, j;

    outimage = (Image*)malloc(sizeof(Image));

    outimage->Type = image->Type;
    if (outimage->Type == GRAY)   size = image->Width * image->Height;
    else if (outimage->Type == COLOR) size = image->Width * image->Height * 3;

    outimage->Width = image->Width;
    outimage->Height = image->Height;
    outimage->num_comment_lines = image->num_comment_lines;

    /*--------------------------------------------------------*/
    /* Copy Comments for Original Image      */
    for (j = 0; j < outimage->num_comment_lines; j++)
        strcpy(outimage->comments[j], image->comments[j]);

    /*----------- Add New Comment  ---------------------------*/
    strcpy(outimage->comments[outimage->num_comment_lines], comment);
    outimage->num_comment_lines++;


    outimage->data = (unsigned char*)malloc(size);
    if (!outimage->data) {
        printf("cannot allocate memory for new image");
        exit(0);
    }
    return(outimage);
}

