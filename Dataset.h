#ifndef DATASET_H
#include "common.h"
#define DATASET_H
using namespace cv;
using namespace std;
class DQT;
class Dataset {
public:
	int point;
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
	vector<Mat> p_images, n_images;
	Mat npdTable;
	public:
	Dataset(char*, char*, char*);
	void readImage(vector<Mat>&, int, char*, int);
	void calculateFea(Mat&, const vector<Mat>&, const int&, int);
	void calculateNpdTable();
	void initWeight(int nPos, int nNeg);
	void initSamples();
    void CalcuWeight();
    void AddNegSam(vector<DQT*>& weakClassifier,int numOfSam);
    void TrimWeight(vector<int>& posIndex, vector<int>& negIndex);
};

namespace selfxNode{
class xNode{
public:
    int ID;
    float leftFit, rightFit, threshold1, threshold2, score, parentFit;
    int featId, level, minLeaf;
    vector<int> pInd, nInd;
    xNode * lChild, *rChild;
    double SplitxNode(Dataset & dataset);
    void Init(float parentFit, int minLeaf_);
    double RecurLearn(Dataset & dataset);
};

}
using namespace selfxNode;

class DQT{
public:
    float threshold;
    float FAR;
    xNode * root;
    void CreateTree(Dataset &dataset,vector<int>& pInd, vector<int> &nInd,
                    int minLeaf);
    void Init_tree(vector<int>& pInd,
                   vector<int>& nInd,
                   int minLeaf);
    void LearnDQT(Dataset & dataset);
    void ReleaseSpace(xNode *node);
    void CalcuThreshold(Dataset &dataset);
    double TestMyself(const cv::Mat& x);
    double RecurTest(const cv::Mat& x, xNode * node);
    //TODO
    void CaucultxNode();
    void SaveTree(char * fileName, int ID);
    vector< xNode*> LinkxNodeToVec();
    void RecurAddxNode(vector< xNode*>& vec, xNode* node);
};

class Adaboost{
public:
    vector<DQT*> weakClassifier;
    void TrainFaceDector(Dataset & dataset);
    void LearnAdaboost(Dataset & dataset);
    void TestAdaboost(vector<double>& Fx, vector<int>& passCount, cv::Mat& X);
};


#endif /* Adaboost_hpp */
