//
//  DQT.hpp
//  NPD_A
//
//  Created by 陈石涛 on 28/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//

#ifndef DQT_h
#include <stdio.h>
#include<vector>
#define DQT_h
using namespace std;

class Node{
public:
    float leftFit, rightFit, threshold1, threshold2, score, parentFit;
    int featId, level;
    vector<int> pInd, nInd;
    Node * lChild, *rChild;
    void SplitNode(DataSet & dataset);
};
class DQT{
public:
    Node * root;
    void Init();
    void LearnDQT(DataSet & dataset);
};
#endif /* DQT_hpp */
