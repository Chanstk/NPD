#ifndef DATASET_H
#include "common.h"
#define DATASET_H
using namespace cv;
using namespace std;
class DQT;
class xNode;
class Dataset {
	public:
		int point;
		//Mat pSam;  //
		//Mat nSam;  //
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
		vector<Mat> p_images, n_images;   //
		vector<Mat> bootImages;
		Mat npdTable;
		vector<double> picSlideNums;
		vector<double> picGridNums;
		vector<int> dim_table;
	public:
		Dataset(char*, char*, char*);

		void SlideImage(vector<Mat> &face, vector<DQT*>&weakClassifier, Mat& fullPic, float winStepScale);

		void readImage(vector<Mat>&, int, char*, int);
		void calculateFea(Mat&, const vector<Mat>&, const int&, int);
		void calculateNpdTable();
		void initWeight(int nPos, int nNeg);
		void initSamples();
		void CalcuWeight();
		unsigned char  GetFeatViaDim(int dim, int index, bool sign);  //get the ith dim feature
		void GetFeat(int index, bool sign, vector<unsigned char> & featList);  //get the whole feature vector of a particular image
		void GetFeatViaDimPerPic(vector<int> & index, bool sign, int dim, vector<unsigned char> &featList);  //get all the features of one dim
		void AddNegSam(vector<DQT*>& weakClassifier,int numOfSam);
		void TrimWeight(vector<int>& posIndex, vector<int>& negIndex);

		int getI(const int dim);
		unsigned char getIthDimensionFeature(const int dim ,const Mat &ima);
		double testDQT(xNode* &root, Mat &win);
		bool testWin(vector<DQT*> &weak, Mat &win);
};

class xNode{
	public:
		int ID;
		double leftFit, rightFit, threshold1, threshold2, score, parentFit;
		int featId, level, minLeaf;
		vector<int> pInd, nInd;
		xNode * lChild, *rChild;
		double SplitxNode(Dataset & dataset);
		void Init(double parentFit, int minLeaf_);
		double RecurLearn(Dataset & dataset);
};


class DQT{
	public:
		double threshold;
		double FAR;
		xNode *root;
		void CreateTree(Dataset &dataset,vector<int>& pInd, vector<int> &nInd,
				int minLeaf);
		void Init_tree(vector<int>& pInd,
				vector<int>& nInd,
				int minLeaf);
		void LearnDQT(Dataset & dataset);
		void ReleaseSpace(xNode *node);
		void CalcuThreshold(Dataset &dataset);
		double TestMyself(vector<unsigned char> &x);
		double RecurTest(vector<unsigned char> & x, xNode * node);
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
		void TestAdaboost(Dataset& dataset, vector<double>& Fx, vector<int>& passCount, bool sign, int num );
};


#endif /* Adaboost_hpp */
