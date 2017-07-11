//
//  Adaboost.hpp
//  NPD_A
//
//  Created by 陈石涛 on 29/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//

#ifndef Adaboost_hpp
#include "common.h"
#include "Dataset.h"
#define Adaboost_hpp
using namespace std;
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
