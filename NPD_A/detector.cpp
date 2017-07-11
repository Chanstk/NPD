#include"Detector.h"
#include"math.h"
#include"pugixml.hpp"
using namespace pugi;

Detector::Detector():npd_table(256, 256, 1) { rect.reserve(10); }

void Detector::clear()
{
	for (int i = 0; i < 512; i++)
		aux_vec[i] = NULL;

	return;
}

int Detetor::getNode(xNode* &root, xml_node& node)
{
	root = new xNode();
	sscanf(node.attribute("ID").value(), "%d", &(root->ID));
	sscanf(node.attribute("leftFit").value(), "%f", &(root->leftFit));
	sscanf(node.attribute("rightFit").value(), "%f", &(root->rightFit));
	sscanf(node.attribute("threshold1").value(), "%f", &(root->threshold1));
	sscanf(node.attribute("threshold2").value(), "%f", &(root->threshold2));
	sscanf(node.attribute("featID").value(), "%d", &(root->featId));

	return root->ID;
}

void Detector::getTree(int i)
{
	if (aux_vec[i]) {
		aux_vec[i]->lChild = aux_vec[2 * i];
		getTree(2 * i);

		aux_vec[i]->rChild = aux_vec[2 * i + 1];
		getTree(2 * i + 1);
	}
	return ;
}

void Detector::loadModel(char *xml_file)
{
	xml_document doc;
	xml_parse_result result = doc.load_file("xml_file");

	for (xml_node tree = doc.child("Tree"); tree; tree = tree.next_sibling("Tree")) {

		DQT *dqt = new DQT();
		sscanf(tree.attribute("threshold").value(), "%f", dqt->threshold);  //get rt
		clear();  //clear the array

		for (xml_node node = tree.child("Node"); node; node = node.next_sibling("Node")) {
			xNode *root;
			aux_vec[ getNode(root, node) ] = root;  //flag
		}
		getTree(1);  //link the whole tree
		dqt->root = aux_vec[1];
		model.push_back(dqt);
	}
	return ;
}

void Detector::getNpdTable()
{
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			double fea = 0.5;
			if (i > 0 || j > 0) fea = double(i) / (i + j); 
			fea = floor(256 * fea);
			if (fea > 255) fea = 255;

			npd_table.at<unsigned char>(i, j) = (uchar) fea;
		}
	}
	return ;
}

void Detector::detect(char* ima_name)
{
	Mat ima = imread(ima_name, CV_LOAD_IMAGE_GRAYSCALE);

	int row = ima.rows;
	int col = ima.cols;
	float scale = 1;
	int k = 0;

	for (; row / scale > 20 && col / scale > 20; k++) {
		scale *= pow(1.1, k);

		Mat tmp;
		resize(ima, tmp, Size(row / scale, col / scale), row / scale, col / scale);

		int t_row = tmp.rows;
		int t_col = tmp.cols;

		for (int i = 0; i <= t_row - 20; i += 2) {
			for (int j = 0; j <= t_col - 20; j += 2) {
				Mat win(20, 20, 1);
				vector<unsigned char> fea;
				fea.reserve(79800);

				getSubwin(tmp, i, j, win);  //win_size
				getFea(win, fea);
				if (testModel(fea)) {
					Rect t_rect(j, i, 20 * scale, 20 * scale);
					rect.push_back(t_rect);
				} else {
					continue;
				}
			}
		}
	}
}

float testDQT(xNode* &root, vector<unsigned char>& fea)
{
	xNode *p = root;
	xNode *q = p;
	unsigned char t_fea;
	while (p) {
		q = p;
		t_fea = fea[ root->featId ];
		p = t_fea > p->threshold1 || t_fea < p->threshold2 ?
			p->lChild : p->rChild;
	}
	return t_fea > q->threshold1 || t_fea < q->threshold2 ?
		q->leftFit : q->rightFit;
}

bool Detector::testModel(vector<unsigned char>& fea)
{
	int T = model.size();
	float score = 0.0;
	int i = 0;
	for (; i < T; i++) {
		score += testDQT(model[i]->root, fea);
		if (score < model[i]->threshold) break;
	}
	return i == T ? true : false;
}

void Detector::getSubwin(Mat& ima, int y, int x, Mat& win)
{
	Rect t_rect = Rect(x, y, 20, 20);  //axis
	ima(t_rect).copyTo(win);

	return ;
}

void Detector::getFea(Mat& win, vector<unsigned char>& fea)
{
	unsigned char *addr = win.data;
	for (int i = 0; i < 400; i++) {
		addr = win.data + i;
		for (int j = i + 1; j < 400; j++) 
			fea.push_back( npd_table.at<unsigned char>(*addr, *(addr - i + j)) );
	}
	return ;
}

void Detector::drawRec(char* ima_name)
{
	Mat ima = imread(ima_name, CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < rect.size(); i++) {
		rectangle(ima, rect[i], Scalar(255, 0, 0));
	}
	cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
	imshow("test", ima);
	waitKey(0);

	return ;
}

extern Parameter para = Parameter();

int main(int argc, char* argv[])
{
	Detector my_det;
	my_det.loadModel(argv[1]);
	my_det.detect(argv[2]);
	my_det.drawRec(argv[2]);
	return 0;
}
