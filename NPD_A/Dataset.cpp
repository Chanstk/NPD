
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

void Dataset::readImage(vector<cv::Mat>& images, int num_of_sams, char *fileName, int neg)
{
    int count = 0;
    ifstream file(fileName);
    string imageName;
    while (getline(file, imageName))
    {
        cv::Mat image = cv::imread(imageName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
        if(image.empty())
        {
            printf("empty file path:%s\n", imageName.c_str());
            continue;
        }
       if(!neg){ 
        if (image.cols != para.obj_size || image.rows != para.obj_size)
        {
            cv::resize(image, image, cv::Size(para.obj_size, para.obj_size));
        }
	   }
        images.push_back(image);
        count++;
		//flip picture
		if(!neg){
			cv::Mat f_image;
			cv::flip(image, f_image, 1);
			images.push_back(f_image);
			count++;
		}

        if(count >= num_of_sams){
            cout<<"The picture has been already enough"<<endl;
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

void Dataset::calculateFea(Mat& sam, const vector<Mat>& images, const int& num,int neg)
{
    int pixels = para.obj_size * para.obj_size;
    sam = Mat(num, pixels * (pixels - 1) / 2, CV_8UC1);  //20 * 20;
	for (int k = 0; k < num; k++) {
		int n = 0;
		uchar* addr;
		Mat img;
		if(!neg)
			//if the images are postive samples
			img = images[k];
		else{
			//extract neg sample randomly
			int rnd = rand() % (int) images.size();
			int x = rand() % (images[rnd].cols - para.obj_size);
			int y = rand() % (images[rnd].rows - para.obj_size);
			img = images[rnd](Rect(x ,y ,para.obj_size, para.obj_size));
/*			string a = "./te/";
			string b = ".jpg";
			string c = std::to_string(k);
			string d = a + c + b;
			imwrite(d.c_str(),img);
			if(k == 1000)
				exit(0);*/
		}
		for (int i = 0; i < pixels; i++) {
            //TODO
			addr = img.data + i;
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
    for(int i = 0; i < nPos; i++){
        pweight.push_back((float)1.0 / nPos);
        posFit.push_back(0);
    }
    for(int i = 0; i < nNeg; i++){
        nweight.push_back((float)1.0 / nNeg);
        negFit.push_back(0);
    }
}

void Dataset::TrimWeight(vector<int>& posIndex, vector<int>& negIndex){
    vector<int> trimedPosIndex;
    vector<int> posIndexSort(posIndex);
    vector<double> posW;
    for(int i = 0; i < lengthOfPosW; i++)
        posW.push_back(pweight[i]);
    //TODO
    IndexSort(posIndexSort, posW);
    
    double cumsum = 0;
    int k = 0;
    for (int i = 0; i < int(posIndexSort.size()); i++) {
        cumsum += pweight[posIndexSort[i]];
        if (cumsum >= para.trimFrac) {
            k = i;
            break;
        }
    }
    k = CHEN_MIN(k, (int)posIndex.size() - para.minSamples);
    
    double trimWeight = pweight[posIndexSort[k]];
    
    for (int i = 0; i < int(posIndex.size()); i++) {
        if (pweight[posIndex[i]] >= trimWeight)
            trimedPosIndex.push_back(posIndex[i]);
    }
    posIndex.clear();
    for(int i = 0; i <(int)trimedPosIndex.size(); i++ )
        posIndex.push_back(trimedPosIndex[i]);
    
    //trim neg weight
    vector<int> trimedNegIndex;
    vector<int> negIndexSort(negIndex);
    vector<double> negW;
    for(int i = 0; i < lengthOfNegW; i++)
        negW.push_back(nweight[i]);
    //TODO
    IndexSort(negIndexSort, negW);
    
    cumsum = 0;
    //int k;
    for (int i = 0; i < int(negIndexSort.size()); i++) {
        cumsum += nweight[negIndexSort[i]];
        if (cumsum >= para.trimFrac) {
            k = i;
            break;
        }
    }
    k = CHEN_MIN(k, (int)negIndex.size() - para.minSamples);
    
    trimWeight = nweight[negIndexSort[k]];
    
    for (int i = 0; i < int(negIndex.size()); i++) {
        if (nweight[negIndex[i]] >= trimWeight)
            trimedNegIndex.push_back(negIndex[i]);
    }
    negIndex.clear();
    for(int i = 0; i <(int)trimedNegIndex.size(); i++ )
        negIndex.push_back(trimedNegIndex[i]);
}

void Dataset::CalcuWeight(){
    //posfit的大小为pSam的大小，只是未通过检测的样本posfit 为0，无意义
    int n = (int)pInd.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        pweight[pInd[i]] = min(exp(-1 * posFit[pInd[i]]), para.maxWeight);
        sum += pweight[pInd[i]];
    }
    if (sum == 0) {
        for (int i = 0; i < n; i++) {
            pweight[pInd[i]] = 1./n;
        }
    }
    else{
        for (int i = 0; i < n; i++) {
            pweight[pInd[i]] /= sum;
        }
    }
    n = (int)nInd.size();
    sum = 0;
    for (int i = 0; i < n; i++) {
        nweight[nInd[i]] = min(exp(1 * negFit[nInd[i]]), para.maxWeight);
        sum += nweight[nInd[i]];
    }
    if (sum == 0) {
        for (int i = 0; i < n; i++) {
            nweight[nInd[i]] = 1./n;
        }
    }
    else{
        for (int i = 0; i < n; i++) {
            nweight[nInd[i]] /= sum;
        }
    }
}

void Dataset::AddNegSam(int numOfSam){
    float a = nInd.size();
    int count = 0;
    int pixels = para.obj_size * para.obj_size;
    vector<int> formNInd(nInd);
    srand((int)time(0));
    for(int i = 0; i < nSam.rows; i++){
        //如果该负样本已经无效
        if(find(nInd.begin(), nInd.end(), i)== nInd.end()){
            //替换原有负样本
            count++;
            int n = 0;
	    //randomseletc neg pic
			int rnd = rand() % (int) n_images.size();
			int x = rand() % (n_images[rnd].cols - para.obj_size);
			int y = rand() % (n_images[rnd].rows - para.obj_size);
		 	cv::Mat im =n_images[rnd](Rect(x, y, para.obj_size, para.obj_size));	
            
			uchar* addr = im.data;
            for (int k = 0; k < pixels; k++) {
                //TODO
                addr = im.data + k;
                for (int j = k + 1; j < pixels; j++) {
                    nSam.at<uchar>(i, n++) = npdTable.at<uchar>(*addr, *(addr - k + j));
                }
            }
            nInd.push_back(i);
            nNeg++;
            nweight[i] = ((float)1) / a;
        }
    }
    for(int i = 0; i <(int) formNInd.size(); i++)
        nweight[formNInd[i]] = a / (a + count);
    float sum = 0;
    for(int i = 0; i <(int) nInd.size(); i++)
        sum += nweight[nInd[i]];
    for(int i = 0; i <(int) nInd.size(); i++)
        nweight[nInd[i]] /= sum;
    cout<<"Bootstrap Done"<<endl;
    return;
}
void Dataset::initSamples()
{
    calculateNpdTable();

	int randomSelect = 1;
    cout<<"Prepare postive samples"<<endl;
    readImage(p_images, para.numPosSample, pfile, 0);
    cout<<"The number of postvie samples is "<<p_images.size()<<endl;

    cout<<"Prepare negtive samples"<<endl;
    readImage(n_images, p_images.size() * para.negRatio, nfile,randomSelect);
    cout<<"The number of negtive samples is "<<n_images.size()<<endl;

    calculateFea(pSam, p_images, p_images.size(),0);
    for(int i = 0; i < (int)pSam.rows; i++)
        pInd.push_back(i);

    calculateFea(nSam, n_images, p_images.size() * para.negRatio,randomSelect);
    for(int i = 0; i < (int)nSam.rows; i++)
        nInd.push_back(i);

    nPos = (int)pInd.size();
    nNeg = (int)nInd.size();
//	initWeight(nPos, pweight);
//	initWeight(nNeg, nweight);
	return ;
}

