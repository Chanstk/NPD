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

extern Parameter para = Parameter();
int main(int argc, const char * argv[]) {
    para.paraDefine();
	srand((int)time(0));
	omp_set_num_threads(6);
    Dataset dataset("pos.txt", "neg.txt", "boot.txt");
    dataset.initSamples();
    Adaboost *adaboost = new Adaboost();
    adaboost->TrainFaceDector(dataset);
    return 0;
}
