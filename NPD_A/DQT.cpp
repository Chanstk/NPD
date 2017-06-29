//
//  DQT.cpp
//  NPD_A
//
//  Created by 陈石涛 on 28/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//

#include "DQT.h"
void DQT::Init(){
    this->root = new Node();
    root->parentFit = 0;
}
void Node::SplitNode(DataSet &dataset){
    double w;
    for(int i = 0; i < pInd.size(); i++)
        w += dataset
}
