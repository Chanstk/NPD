
#include"Dataset.h"
using namespace std;
using namespace cv;
typedef unsigned char  uchar;
extern Parameter para;
Dataset::Dataset(char *pf, char *nf):npdTable(256, 256, 1)
{
	nPos = 0;
	nNeg = 0;
	pfile = pf;
	pfile = nf;
	p_images.reserve(10000);
	n_images.reserve(20000);
	return ;
}

void Dataset::readImage(vector<Mat>& images, int num_of_sams, char *file)
{
	ifstream inf(file);
	string name;
	while(getline(inf, name)) {
		num_of_sams++;
		Mat gray = imread(name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		images.push_back(gray);
	}
	inf.close();

	return ;
}

void Dataset::calculateNpdTable()
{
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			double fea = 0.5;
			if (i > 0 || j > 0) fea = double(i) / (i + j);
			fea = floor(256 * fea);
			if (fea > 255) fea = 255;

			npdTable.at<uchar>(i, j) = (uchar) fea;
		}
	}
	return ;
}

void Dataset::calculateFea(Mat& sam, const vector<Mat>& images, const int& num)
{
	int pixels = 400;  //20 * 20;
	for (int k = 0; k < num; k++) {
		int n = 0;
		uchar* addr = (images[k]).data;
		for (int i = 0; i < pixels; i++) {
			addr = images[i].data + i;
			for (int j = i + 1; j < pixels; j++) {
				sam.at<uchar>(k, n++) = npdTable.at<uchar>(*addr, *(addr - i + j));
			}
		}
	}
	return ;
}

void Dataset::initWeight(int nPos, int nNeg)
{
    posFit.clear();
    negFit.clear();
    this->pweight = new float[nPos];
    this->nweight = new float[nNeg];
    pInd.clear();
    for(int i = 0; i < nPos; i++){
        pweight[i] = ((float)1.0)/ nPos;
        pInd.push_back(i);
        posFit.push_back(0);
    }
    
    for(int i = 0; i < nNeg; i++){
        nweight[i] = ((float)1.0)/ nNeg;
        nInd.push_back(i);
        negFit.push_back(0);
    }
}

void Dataset::TrimWeight(vector<int>& posIndex, vector<int>& negIndex, Dataset &dataset){
    vector<int> trimedPosIndex;
    vector<int> posIndexSort(posIndex);
    //TODO
    IndexSort(posIndexSort, posW);
    
    double cumsum = 0;
    int k = 0;
    for (int i = 0; i < int(posIndexSort.size()); i++) {
        cumsum += dataset.pweight[posIndexSort[i]];
        if (cumsum >= para.trimFrac) {
            k = i;
            break;
        }
    }
    k = min(k, posIndex.size() - para.minSamples);
    
    double trimWeight = dataset.pweight[posIndexSort[k]];
    
    for (int i = 0; i < int(posIndex.size()); i++) {
        if (dataset.pweight[posIndex[i]] >= trimWeight)
            trimedPosIndex.push_back(posIndex[i]);
    }
    posIndex.clear();
    for(int i = 0; i <(int)trimedPosIndex.size(); i++ )
        posIndex.push_back(trimedPosIndex[i]);
    
    //trim neg weight
    vector<int> trimedNegIndex;
    vector<int> negIndexSort(negIndex);
    //TODO
    IndexSort(negIndexSort, negW);
    
    cumsum = 0;
    //int k;
    for (int i = 0; i < int(negIndexSort); i++) {
        cumsum += dataset.nweight[negIndexSort[i]];
        if (cumsum >= para.trimFrac) {
            k = i;
            break;
        }
    }
    k = min(k, negIndex.size() - minSamples);
    
    trimWeight = dataset.nweight[negIndexSort[k]];
    
    for (int i = 0; i < int(negIndex.size()); i++) {
        if (dataset.nweight[negIndex[i]] >= trimWeight)
            trimedNegIndex.push_back(negIndex[i]);
    }
    negIndex.clear();
    for(int i = 0; i <(int)trimedNegIndex.size(); i++ )
        negIndex.push_back(trimedNegIndex[i]);
}

void Dataset::CalcuWeight(){
    int n = this->posFit.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        pweight[this->pInd[i]] = min(exp(-1 * this->posFit[i]), para.maxWeight);
        sum += pweight[this->pInd[i]];
    }
    if (sum == 0) {
        for (int i = 0; i < n; i++) {
            pweight[this->pInd[i]] = 1./n;
        }
    }
    else{
        for (int i = 0; i < n; i++) {
            pweight[this->pInd[i]] /= sum;
        }
    }
    n = this->negFit.size();
    sum = 0;
    for (int i = 0; i < n; i++) {
        nweight[this->nInd[i]] = min(exp(1 * this->negFit[i]), para.maxWeight);
        sum += nweight[this->nInd[i]];
    }
    if (sum == 0) {
        for (int i = 0; i < n; i++) {
            nweight[this->nInd[i]] = 1./n;
        }
    }
    else{
        for (int i = 0; i < n; i++) {
            nweight[this->nInd[i]] /= sum;
        }
    }
}
void Dataset::initSamples()
{
	calculateNpdTable();
	readImage(p_images, nPos, pfile);
	readImage(n_images, nNeg, nfile);
	calculateFea(pSam, p_images, nPos);
	calculateFea(nSam, n_images, nNeg);
	initWeight(nPos, pweight);
	initWeight(nNeg, nweight);
	return ;
}
