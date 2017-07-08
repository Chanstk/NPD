//
//  Adaboost.cpp
//  NPD_A
//
//  Created by 陈石涛 on 29/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//

#include "Adaboost.h"
using namespace std;
extern Parameter para;
vector<int> *s_index;
vector<double> *s_content;
bool IndexSorterCompareUp(int i, int j) {
    return ((s_content->at(i)) < (s_content->at(j)));
}

bool IndexSorterCompareDown(int i, int j) {
    return ((s_content->at(i)) > (s_content->at(j)));
}

void IndexSort(vector<int>& _index, vector<double>& _content, int direction) {
    s_index = &_index;
    s_content = &_content;
    if (direction > 0)
    {
        sort(s_index->begin(), s_index->end(), IndexSorterCompareUp);
    }
    else
    {
        sort(s_index->begin(), s_index->end(), IndexSorterCompareDown);
    }
};
void Adaboost::TrainFaceDector(Dataset &dataset){
    //参数设定
    //提取正负样本
    //提取特征值
    bool finished = false;
    int finalNegs = 0;
    int T = 0;
    int numFaces = dataset.nPos;
    int desiredNumNegs = numFaces * para.negRatio;
    while(true){
        //TOOD
        //负样本bootstrap
        int needNumNegs = desiredNumNegs - dataset.nNeg;
        if(needNumNegs > 0)
            dataset.AddNegSam(needNumNegs);
        //负样本不够，退出
        if(dataset.nNeg < para.finalNegs)
            break;
        T = (int)weakClassifier.size();
        if(dataset.nNeg < para.finalNegs){
            printf("\n\nNo enough negative examples to bootstrap (nNeg=%d). The detector training is terminated.\n", dataset.nNeg);
            break;
        }
        
        if (dataset.nNeg == desiredNumNegs)
            LearnAdaboost(dataset);
        else{
            para.negRatio = finalNegs / dataset.nSam.rows;
            LearnAdaboost(dataset);
            finished = true;
        }
        
        if(weakClassifier.size() == T){
            cout<<("\n\nNo effective features for further detector learning.\n");
            break;
        }
        
        T = (int)weakClassifier.size();
        
        //保存模型
        
        
        double far = 1;
        for (int i = 0; i < int(weakClassifier.size()); i++) {
            far *= weakClassifier[i]->FAR;
        }
        cout<<"Weak classifier " << T <<" FAR: "<<far;
        
        if (far <= para.MAXFAR || T >= para.max_stage || finished) {
            printf("\n\nThe detector training is finished.\n");
            break;
        }
    }
}
void Adaboost::TestAdaboost(vector<double>& Fx, vector<int>& passCount, cv::Mat& X){
    int n = X.rows;
    Fx.clear();
    passCount.clear();
    Fx.resize(n);
    passCount.resize(n);
    for (int i = 0; i < n; i++) {	//for each image
        bool run = true;
        
        for (int j = 0; j < int(weakClassifier.size()) && run; j++) {	//for each weak classifier
            
            double fx = weakClassifier[j]->TestMyself(X.row(i));
            Fx[i] += fx;
            if (Fx[i] >= weakClassifier[j]->threshold) {
                passCount[i]++;
            }
            else {
                run = false;
            }
        }
    }
}
void Adaboost::LearnAdaboost(Dataset &dataset){
    int nPos = dataset.pSam.rows;
    int nNeg = dataset.nSam.rows;
    
    vector<double> posFit, negFit;
    vector<int> passCount;
    int T = (int)weakClassifier.size();
    if(T){
        cout<<"Test current model"<<endl;
        //测试样本， dataset里面将未经过测试的样本剔除
        //测试正样本
        dataset.pInd.clear();
        dataset.posFit.clear();
        dataset.posFit.resize(nPos);
        TestAdaboost(posFit, passCount, dataset.pSam);
        for (int i = 0; i < int(passCount.size()); i++) {
            if (passCount[i] == T) {
                //通过测试的正样本
                dataset.pInd.push_back(i);
                dataset.posFit[i] = posFit[i];
            }
        }
        dataset.nPos = (int)dataset.pInd.size();
        if (dataset.pInd.size() < nPos) {
            cout << "Warning: some positive samples cannot pass all stages. pass rate is "
            << ((dataset.pInd.size())/(double)(nPos)) << endl;
        }

        //测试负样本
        dataset.nInd.clear();
        dataset.negFit.clear();
        dataset.negFit.resize(nNeg);
        TestAdaboost(negFit, passCount, dataset.nSam);
        for (int i = 0; i < int(passCount.size()); i++) {
            if (passCount[i] == T) {
                dataset.nInd.push_back(i);
                dataset.negFit[i] = negFit[i];
            }
        }
        dataset.nNeg = (int)dataset.nInd.size();
        if (nNeg > dataset.nInd.size()) {
            cout << "Warning: some negative samples cannot pass all stages, pass rate is "
            << ((dataset.nInd.size())/(double)(nNeg)) << endl;
        }
        
        //重新计算样本权值
        dataset.CalcuWeight();
        dataset.CalcuWeight();
            
    }else
        dataset.initWeight(nPos, nNeg);
    
    int nNegPass = dataset.nNeg;
    for(int t = T; t < para.max_stage; t++){
        if(dataset.nNeg < para.minSamples){
            cout << endl << "No enough negative samples. The Adaboost learning terninates at iteration "
            << t << ". nNegPass = " << dataset.nNeg << endl;
            break;
        }
        
        //选取部分正负样本作为根节点的输入样本。
        int nPosSam = max((int)round(dataset.nPos* para.samFrac), para.minSamples);
        vector<int> posIndex(dataset.nPos);
        for (int i = 0; i < dataset.nPos; i++)
            posIndex[i] = dataset.pInd[i];
        random_shuffle(posIndex.begin(), posIndex.end());
        //posIndex 为输入样本，输入样本只是dataset.pInd的一部分
        posIndex.resize(nPosSam);
        
        int nNegSam = max((int)round(dataset.nNeg * para.samFrac), para.minSamples);
        vector<int> negIndex(dataset.nNeg);
        for (int i = 0; i < dataset.nNeg; i++)
            negIndex[i] = dataset.nInd[i];
        random_shuffle(negIndex.begin(), negIndex.end());
        negIndex.resize(nNegSam);
        //TODO
        dataset.TrimWeight(posIndex, negIndex);
        nPosSam = (int)posIndex.size();
        nNegSam = (int)negIndex.size();
        
        int minLeaf_t = max( (int)round((nPosSam+nNegSam)* para.minLeafFrac), para.minLeaf);
        
        
        printf("\nIter %d: nPos=%d, nNeg=%d, ", t, nPosSam, nNegSam);
        
        DQT *tree = new DQT();
        tree->CreateTree(dataset,posIndex, negIndex, minLeaf_t);
        if(tree->root->featId == -1){
            printf("\n\nNo available features to satisfy the split. The AdaBoost learning terminates.\n");
            break;
        }
        tree->CalcuThreshold(dataset);
        vector<int> temNegPassIndex;
        for (int i = 0; i < int(dataset.nNeg); i++){
            dataset.negFit[dataset.nInd[i]] += tree->TestMyself(dataset.nSam.row(dataset.nInd[i]));
            if(dataset.negFit[dataset.nInd[i]] >= tree->threshold)
                temNegPassIndex.push_back(dataset.nInd[i]);
        }
        
        dataset.nInd.swap(temNegPassIndex);
        dataset.nNeg = (int)dataset.nInd.size();
        tree->FAR = (float)(dataset.nNeg * 1.0 / nNegPass);
        nNegPass = dataset.nNeg;
        tree->SaveTree(para.modelName, t);
        weakClassifier.push_back(tree);
        double FAR = 1;
        for (int i = 0; i < weakClassifier.size(); i++)
            FAR *= weakClassifier[i]->FAR;
        
        cout<<"FAR at " <<t<<" stage is "<<FAR * 100<<"%"<<endl;
        if (FAR <= para.MAXFAR) {
            printf("\n\nThe training is converged at iteration %d. FAR = %.2f%%\n", t, FAR * 100);
            break;
        }
        if (nNegPass < dataset.nNeg * para.minNegRatio || nNegPass < para.minSamples) {
            printf("\n\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n", t, nNegPass);
            break;
        }
        dataset.CalcuWeight();
    }
    cout<<"The adaboost training is finished at inner cycle."<<endl;
}


