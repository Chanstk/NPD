//
//  main.cpp
//  NPD_A
//
//  Created by 陈石涛 on 28/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//
#include "common.h"
#include "Dataset.h"
#include "Adaboost.h"
extern Parameter para;
int main(int argc, const char * argv[]) {
    // insert code here...
    para.paraDefine();
    Dataset dataset("pos.txt", "neg.txt", "boot.txt");
    dataset.initSamples();
//    Adaboost adaboost = new Adaboost();
//    adaboost->TrainFaceDector(dataset());
    
    return 0;
}
