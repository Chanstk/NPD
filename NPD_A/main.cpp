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
    cout<<"The max thread is "<<omp_get_max_threads()<<endl;
    omp_set_num_threads(4);
    cout<<"The current thread number is "<<omp_get_num_threads()<<endl;
#pragma omp parallel
    {
        cout<<"Hello "<<"I am Thread"<<omp_get_thread_num()<<endl;
    }
    exit(0);
    para.paraDefine();
    Dataset dataset("pos.txt", "neg.txt", "boot.txt");
    dataset.initSamples();
    cout<<(int) dataset.p_images.size()<<endl;
    cout<<(int) dataset.n_images.size()<<endl;
    cout<<(int) dataset.bootStrapImages.size()<<endl;
    cout<<(int) dataset.nPos<<endl;
    cout<<(int) dataset.nNeg<<endl;
    Adaboost *adaboost = new Adaboost();
    adaboost->TrainFaceDector(dataset);
    
    return 0;
}
