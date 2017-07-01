//
//  DQT.cpp
//  NPD_A
//
//  Created by 陈石涛 on 28/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//
#include "Adaboost.h"
extern Parameter para;

void WeightHist(const cv::Mat& X, float* W, vector<int>& index, int n, int count[256], double wHist[256]){
    memset(wHist, 0, 256 * sizeof(double));
    
    for (int j = 0; j < n; j++){
        unsigned char bin = X.ptr(index[j])[0];
        count[bin]++;
        wHist[bin] += W[index[j]];
    }
}

void DQT::Init_tree(vector<int>& pInd,
          vector<int>& nInd,
          int minLeaf){
    this->root = new Node();
    for(int i = 0; i < pInd.size(); i++)
        root->pInd.push_back(pInd[i]);
    for(int i = 0; i < nInd.size(); i++)
        root->pInd.push_back(nInd[i]);
    root->Init(0, minLeaf);
    root->level = 1;
}

double Node::SplitNode(Dataset &dataset){
    double w;
    int nPos = pInd.size();
    int nNeg = nInd.size();
    
    for(int i = 0; i < nPos; i++)
        w += dataset.pweight[pInd[i]];
    double minCost = w * (parentFit - 1) * (parentFit - 1);
    
    w = 0;
    for(int i = 0; i < nNeg; i++)
        w += dataset.nweight[nInd[i]];
    
    minCost += w * (parentFit + 1) * (parentFit + 1);
    if (nPos == 0 || nNeg == 0 || nPos + nNeg < 2 * minLeaf)
        return minCost;
    int feaDims = (int)dataset.pSam.cols;
    
    minCost = 1e16f;
    //遍历特征
    for(int i = 0; i < feaDims; i++){
        int count[256];
        double posWhist[256];
        double negWhist[256];
        
        memset(count, 0, 256 * sizeof(int));
        
        WeightHist(dataset.pSam.col(i), dataset.pweight, pInd, nPos, count, posWhist);
        WeightHist(dataset.nSam.col(i), dataset.nweight, nInd, nNeg, count, negWhist);
        
        double posWSum = 0;
        double negWSum = 0;
        
        for (int bin = 0; bin < 256; bin++)
        {
            posWSum += posWhist[bin];
            negWSum += negWhist[bin];
        }
        
        int totalCount = nPos + nNeg;
        
        double minMSE = 3.4e38f;
        int thr0 = -1, thr1 = -1;
        double fit0 = 0, fit1 = 0;
        //左阈值
        for(int v = 0; v < 256; v++){
            
            int rightCount = 0 ;
            double rightPosW = 0;
            double rightNegW = 0;
            //右阈值
            for(int u = v; u < 256; u++){
                rightCount += count[u];
                rightPosW += posWhist[u];
                rightNegW += negWhist[u];
                //右子树样本需要达到特定数量
                if(rightCount < minLeaf) continue;
                
                int leftCount = totalCount - rightCount;
                if(leftCount < minLeaf) break;
                
                
                double leftPosW = posWSum - rightPosW;
                double leftNegW = negWSum - rightNegW;
                
                double leftFit, rightFit;
                
                if (leftPosW + leftNegW <= 0) leftFit = 0.0f;
                else leftFit = (leftPosW - leftNegW) / (leftPosW + leftNegW);
                
                if (rightPosW + rightNegW <= 0) rightFit = 0.0f;
                else rightFit = (rightPosW - rightNegW) / (rightPosW + rightNegW);
                
                double leftMSE = leftPosW * (leftFit - 1) * (leftFit - 1) + leftNegW * (leftFit + 1) * (leftFit + 1);
                double rightMSE = rightPosW * (rightFit - 1) * (rightFit - 1) + rightNegW * (rightFit + 1) * (rightFit + 1);
                
                double mse = leftMSE + rightMSE;
                
                if (mse < minMSE)
                {
                    minMSE = mse;
                    thr0 = v;
                    thr1 = u;
                    fit0 = leftFit;
                    fit1 = rightFit;
                }
            }
        }
        if(thr0 == -1) continue;
        if (minMSE <= minCost)
        {
            minCost = minMSE;
            featId = i;
            threshold1 = thr0;
            threshold2 = thr1;
            leftFit = fit0;
            rightFit = fit1;
        }
    }
    return minCost;
}

double Node::RecurLearn(Dataset & dataset){
    double minCost = this->SplitNode(dataset);
    int nPos = this->pInd.size();
    int nNeg = this->nInd.size();
    //未选择特征
    if(this->featId == -1) return minCost;
    //达到最大树高
    if(this->level >= 8) return minCost;
    
    int lPos = 0, rPos = 0, lNeg = 0, rNeg = 0;
    float leftThr = this->threshold1;
    float rightThr = this->threshold2;
    
    //左右子树初始化
    Node* lChild = new Node();
    Node* rChild = new Node();
    
    lChild->Init(this->leftFit, minLeaf);
    lChild->level = this->level + 1;
    rChild->Init(this->rightFit, minLeaf);
    rChild->level = this->level + 1;
    
    //正样本分入左右子树
    for(int j = 0; j < nPos; j++)
        if(dataset.pSam.at<uchar>(size_t(this->pInd[j]), size_t(this->featId))< leftThr || dataset.pSam.at<uchar>(size_t(this->pInd[j]), size_t(this->featId)) > rightThr)
            lChild->pInd.push_back(this->pInd[j]);
        else
            rChild->pInd.push_back(this->pInd[j]);
    
    //负样本分入左右子树
    for(int j = 0; j < nPos; j++)
        if(dataset.nSam.at<uchar>(size_t(this->nInd[j]), size_t(this->featId))< leftThr || dataset.nSam.at<uchar>(size_t(this->nInd[j]), size_t(this->featId)) > rightThr)
            lChild->nInd.push_back(this->nInd[j]);
        else
            rChild->nInd.push_back(this->nInd[j]);
    
    //左右子树递归
    double minCost1 = lChild->RecurLearn(dataset);
    double minCost2 = rChild->RecurLearn(dataset);
    
    //本节点无左右子树
    if(lChild->threshold1 == -1 && rChild->threshold1 == -1){
        delete lChild;
        delete rChild;
        return minCost;
    }
    //左右子树minCost之和大于本节点,丢弃左右子树
    if (minCost1 + minCost2 >= minCost){
        delete lChild;
        delete rChild;
        return minCost;
    }
    
    minCost = minCost1 + minCost2;
    
    this->lChild = lChild;
    this->rChild = rChild;
    return minCost;
}

void DQT::LearnDQT(Dataset &dataset){
    this->root->RecurLearn(dataset);
}

void DQT::CreateTree(Dataset &dataset,vector<int>& pInd, vector<int> &nInd
                    int minLeaf){
    this->Init_tree(pInd, nInd, minLeaf);
    this->LearnDQT(dataset);
}
void DQT::ReleaseSpace(Node *node){
    if(node->lChild == NULL && node->rChild == NULL){
        delete node;
        return ;
    }
    if(node->lChild ==NULL){
        return ReleaseSpace(node->rChild);
    }
    if(node->rChild ==NULL){
        return ReleaseSpace(node->lChild);
    }
    delete node;
    return ;
}

void Node::Init(float parentFit, int minLeaf_){
    this->parentFit = parentFit;
    featId = -1;
    threshold1 = -1;
    threshold2 = -1;
    minLeaf = minLeaf_;
    lChild = NULL;
    rChild = NULL;
}

//posFx 的值是前n个分类器输出之和
void DQT::CalcuThreshold(Dataset &dataset){
    vector<double> v;
    for (int i = 0; i < int(dataset.nPos); i++) {
        dataset.posFit[dataset.pInd[i]] += this->TestMyself(dataset.pSam.row(dataset.pInd[i]))
        v.push_back(dataset.posFit[dataset.pInd[i]]);
    }
    sort(v.begin(), v.end());
    int index = max((int)floor(dataset.nPos*(1- para.MINDR)), 0)
    this->threshold = v[index];
}

double DQT::RecurTest(const cv::Mat& x, Node * node){
    unsigned char * ptr = x.data
    if(ptr[node->featID] < node->threshold1 || ptr[node->featID] > node->threshold1){
        if(node->lchild == NULL)
            return node->leftFit;
        else
            return RecurTest(x, node->lChild);
    }
    else{
        if(node->rchild == NULL)
            return node->rightFit;
        else
            return RecurTest(x, node->rChild);
    }
}

double DQT::TestMyself(const cv::Mat& x){
    return RecurTest(x, this->root);
}
