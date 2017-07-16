#ifndef DATASET_H
#include "common.h"
#define DATASET_H
using namespace cv;
using namespace std;
class Dataset {
public:
	Mat pSam;
	Mat nSam;
    Mat negPool;
	int nPos;
	int nNeg;
    int bootNum;
    vector<int> pInd, nInd;
	vector<float> pweight;
	vector<float> nweight;
    int lengthOfPosW;
    int lengthOfNegW;
    vector<double> posFit;
    vector<double> negFit;
	char *pfile;
	char *nfile;
    char *bootFile;
	vector<Mat> p_images, n_images, bootStrapImages;
	Mat npdTable;
	public:
	Dataset(char*, char*, char*);
	void readImage(vector<Mat>&, int, char*,int);
	void calculateFea(Mat&, const vector<Mat>&, const int&);
	void calculateNpdTable();
	void initWeight(int nPos, int nNeg);
	void initSamples();
    void CalcuWeight();
    void AddNegSam(int numOfSam);
    void TrimWeight(vector<int>& posIndex, vector<int>& negIndex);
};
#endif
