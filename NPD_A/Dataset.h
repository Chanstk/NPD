#ifndef DATASET_H
#include "common.h"
#define DATASET_H
using namespace cv;
using namespace std;
class Dataset {
public:
	Mat pSam;
	Mat nSam;
	int nPos;
	int nNeg;
    vector<int> pInd, nInd;
	float *pweight;
	float *nweight;
    vector<double> posFit;
    vector<double> negFit;
	char *pfile;
	char *nfile;
	vector<Mat> p_images, n_images;
	Mat npdTable;
	public:
	Dataset(char*, char*);
	void readImage(vector<Mat>&, int, char*);
	void calculateFea(Mat&, const vector<Mat>&, const int&);
	void calculateNpdTable();
	void initWeight(int nPos, int nNeg);
	void initSamples();
    void CalcuWeight();
    void TrimWeight(vector<int>& posIndex, vector<int>& negIndex, Dataset &dataset);
};
#endif
