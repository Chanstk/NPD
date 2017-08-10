#include"Parameter.h"

void Parameter::paraDefine(){
    MINDR = 1;
    MAXFAR = 1e-6;
    tree_level = 6;
    max_stage = 1000;
    obj_size = 20;
    finalNegs = 100;
    minSamples = 100;
    negRatio = 1; 
    minNegRatio = 0.2;
    trimFrac = 0.01;		// weight trimming in AdaBoost
    samFrac = 1;			// the fraction of samples randomly selected in each iteration
	featDim = obj_size * obj_size*(obj_size * obj_size - 1) / 2;
    // for training; could be used to avoid overfitting.
    minLeafFrac = 0.01;		// minimal sample fraction w.r.t.the total number of
    // samples required in each leaf node.This is used to avoid overfitting.
    minLeaf = 100;			// minimal samples required in each leaf node.This is used to avoid overfitting.
    maxWeight = 100;		// maximal sample weight in AdaBoost; used to ensure numerical stability.
    numThreads = 10;		// the number of computing threads in tree learning
    numPosSample = 25000;
    modelName = "Tree.xml";
	bootNum = 10000;
	pugi::xml_document doc;
	doc.load_file(modelName);
	pugi::xml_node tree = doc.append_child("Para");
	tree.append_attribute("MINDR") = MINDR;
	tree.append_attribute("MAXFAR") = MAXFAR;
	tree.append_attribute("treeLevel") = tree_level;
	tree.append_attribute("minSamples") = minSamples;
	tree.append_attribute("minLeaf") = minLeaf;
	tree.append_attribute("negRatio") = negRatio;
	tree.append_attribute("minNegRatio") = minNegRatio;
	tree.append_attribute("numPosSample") = numPosSample;
	tree.append_attribute("minLeafFrac") = minLeafFrac;
	doc.save_file(modelName);
}

