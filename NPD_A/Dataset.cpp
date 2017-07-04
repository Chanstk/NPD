
#include"Dataset.h"
using namespace std;
using namespace cv;
#define CHEN_MIN(i,j)   (((i) > (j)) ? (j) : (i))
#define CHEN_MAX(i,j)   (((i) < (j)) ? (j) : (i))
typedef unsigned char  uchar;
extern Parameter para;
Dataset::Dataset(char *pf, char *nf, char *bf):npdTable(256, 256, 1)
{
	nPos = 0;
	nNeg = 0;
	pfile = pf;
	nfile = nf;
    bootFile = bf;
	return ;
}

void Dataset::readImage(vector<cv::Mat>& images, int num_of_sams, char *fileName)
{
    int count = 0;
    ifstream file(fileName);
    string imageName;
    while (getline(file, imageName))
    {
        cv::Mat image = cv::imread(imageName.c_str(),0);
        if(image.empty())
        {
            printf("empty file path:%s\n", imageName.c_str());
            continue;
        }
        
        if (image.cols != para.windSize || image.rows != para.windSize)
        {
            cv::resize(image, image, cv::Size(para.windSize, para.windSize));
        }
        images.push_back(image);
        count++;
        if(count >= num_of_sams){
            cout<<"The picture has been already enough";
            break;
        }
    }
    file.close();
    printf(" done\n");
    return;
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
    int pixels = para.windSize * para.windSize;
    sam = Mat(num, pixels * (pixels - 1) / 2, CV_8UC1);  //20 * 20;
	for (int k = 0; k < num; k++) {
		int n = 0;
		uchar* addr = (images[k]).data;
		for (int i = 0; i < pixels; i++) {
            //TODO
			addr = images[k].data + i;
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
    this->pweight.clear();
    this->nweight.clear();
    lengthOfPosW = nPos;
    lengthOfNegW = nNeg;
    pInd.clear();
    for(int i = 0; i < nPos; i++){
        pweight.push_back((float)1.0 / nPos);
        posFit.push_back(0);
    }
    nInd.clear();
    for(int i = 0; i < nNeg; i++){
        nweight.push_back((float)1.0 / nNeg);
        negFit.push_back(0);
    }
}

void Dataset::TrimWeight(vector<int>& posIndex, vector<int>& negIndex, Dataset &dataset){
    vector<int> trimedPosIndex;
    vector<int> posIndexSort(posIndex);
    vector<double> posW;
    for(int i = 0; i < dataset.lengthOfPosW; i++)
        posW.push_back(pweight[i]);
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
    k = CHEN_MIN(k, (int)posIndex.size() - para.minSamples);
    
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
    vector<double> negW;
    for(int i = 0; i < dataset.lengthOfNegW; i++)
        negW.push_back(nweight[i]);
    //TODO
    IndexSort(negIndexSort, negW);
    
    cumsum = 0;
    //int k;
    for (int i = 0; i < int(negIndexSort.size()); i++) {
        cumsum += dataset.nweight[negIndexSort[i]];
        if (cumsum >= para.trimFrac) {
            k = i;
            break;
        }
    }
    k = CHEN_MIN(k, (int)negIndex.size() - para.minSamples);
    
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
    int n = (int)this->posFit.size();
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
    n = (int)this->negFit.size();
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

void Dataset::AddNegSam(int numOfSam){
    if(bootStrapImages.size() == 0){
        cout<<"not enough images for bootstrap"<<endl;
        return;
    }
    float a = nInd.size();
    int count = 0;
    int pixels = para.windSize * para.windSize;
    vector<int> formNInd(nInd);
    for(int i = 0; i < nInd.size() + numOfSam; i++){
        if(find(nInd.begin(), nInd.end(), i)!= nInd.end()){
            //替换原有负样本
            count++;
            int n = 0;
            uchar* addr = (bootStrapImages[0]).data;
            for (int k = 0; k < pixels; k++) {
                //TODO
                addr = bootStrapImages[0].data + k;
                for (int j = k + 1; j < pixels; j++) {
                    nSam.at<uchar>(i, n++) = npdTable.at<uchar>(*addr, *(addr - k + j));
                }
            }
            bootStrapImages.erase(bootStrapImages.begin());
            nInd.push_back(i);
            nNeg++;
            nweight[i] = (float)1 / a;
        }
        if(bootStrapImages.size() == 0){
            cout<<"not enough images for bootstrap"<<endl;
            break;
        }
    }
    for(int i = 0; i < formNInd.size(); i++)
        nweight[formNInd[i]] = a / (a + count);
    float sum = 0;
    for(int i = 0; i < nInd.size(); i++)
        sum += nweight[nInd[i]];
    for(int i = 0; i < nInd.size(); i++)
        nweight[nInd[i]] /= sum;
    cout<<"Done"<<endl;
    return;
    
}
void Dataset::initSamples()
{
	calculateNpdTable();
    cout<<"Prepare postive sample"<<endl;
	readImage(p_images, para.numPosSample, pfile);
    cout<<"Prepare negtive sample"<<endl;
	readImage(n_images, para.numPosSample * para.negRatio, nfile);
    cout<<"Prepare bootstrap sample"<<endl;
    readImage(bootStrapImages, para.bootNum, bootFile);
	calculateFea(pSam, p_images, para.numPosSample);
    for(int i = 0; i < (int)pSam.rows; i++)
        pInd.push_back(i);
    calculateFea(nSam, n_images, para.numPosSample * para.negRatio);
    for(int i = 0; i < (int)nSam.rows; i++)
        nInd.push_back(i);
    nPos = (int)pInd.size();
    nNeg = (int)nInd.size();
//	initWeight(nPos, pweight);
//	initWeight(nNeg, nweight);
	return ;
}
