#ifndef DATASET_H
#include "common.h"
#define DATASET_H
using namespace cv;

class Dataset {
	public:
	Mat pSam;
	Mat nSam;;
	int nPos;
	int nNeg;
	float *pweight;
	float *nweight;
	char *pfile;
	char *nfile;
	vector<Mat> p_images, n_images;
	Mat npdTable;

	public:
	Dataset(char*, char*);
	void readImage(vector<Mat>&, int, char*);
	void calculateFea(Mat&, const vector<Mat>&, const int&);
	void calculateNpdTable();
	void initWeight(const int&, float *weight);
	void initSamples();
};
#endif
