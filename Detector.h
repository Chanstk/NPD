#ifndef D_H
#define D_H

#include"Adaboost.h"
#include"common.h"
class Detector {
	public:
	vector<DQT*> model;
	xNode *aux_vec[512];
	Mat npd_table;
	vector<Rect> rect;

	Detector();
	void clear();
	void detect(char*);
	void getTree(int );
	void loadModel(char*);
	void getNpdTable();
	bool testModel(vector<unsigned char>&);
	void getSubwin(Mat&, int, int, Mat&);
	void getFea(Mat&, vector<unsigned char>&);
	void drawRec(char*);
	void getNode(xNode* &, xml_node&);
};
#endif
