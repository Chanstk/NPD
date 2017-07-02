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
int main(int argc, const char * argv[]) {
    // insert code here...
    extern Parameter para();
    para.paraDefine();
    Dataset dataset();
    dataset.initSamples();
    Adaboost adaboost = new Adaboost();
    adaboost->TrainFaceDector(dataset());
    std::cout << "Hello, World!\n";
    return 0;
}
