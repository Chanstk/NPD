//
//  DQT.hpp
//  NPD_A
//
//  Created by 陈石涛 on 28/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//

#ifndef DQT_h
#include "common.h"
#include "Dataset.h"
#define DQT_h
using namespace std;

class Node{
public:
    float leftFit, rightFit, threshold1, threshold2, score, parentFit;
    int featId, level, minLeaf;
    vector<int> pInd, nInd;
    Node * lChild, *rChild;
    double SplitNode(Dataset & dataset);
    void Init(float parentFit);
    double RecurLearn(Dataset & dataset);
};
class DQT{
public:
    Node * root;
    Node * GetTree(Dataset &dataset);
    void Init();
    void LearnDQT(Dataset & dataset);
};
#endif /* DQT_hpp */
