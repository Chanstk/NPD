
#include"Dataset.h"
using namespace std;
using namespace cv;

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

void Dataset::initWeight(const int& num, float *weight)
{
	weight = new float[num];
	for (int i = 0; i < num; i++) {
		weight[i] = 0.5 / num;
	}
	return ;
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
