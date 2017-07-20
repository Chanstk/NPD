#include"Parameter.h"

void Parameter::paraDefine(){
    MINDR =  0.98;
    MAXFAR = 1e-7;
    tree_level = 4;
    max_stage = 1000;
    obj_size = 20;
    finalNegs = 2000;
    minSamples = 2300;
    negRatio =3.5; 
    minNegRatio = 0.3;
    trimFrac = 0.05;		// weight trimming in AdaBoost
    samFrac = 1.0;			// the fraction of samples randomly selected in each iteration
    // for training; could be used to avoid overfitting.
    minLeafFrac = 0.05;		// minimal sample fraction w.r.t.the total number of
    // samples required in each leaf node.This is used to avoid overfitting.
    minLeaf = 100;			// minimal samples required in each leaf node.This is used to avoid overfitting.
    maxWeight = 100;		// maximal sample weight in AdaBoost; used to ensure numerical stability.
    numThreads = 10;		// the number of computing threads in tree learning
    numPosSample = 28000;
    modelName = "Tree.xml";
	bootNum = 10000;
	pugi::xml_document doc;
	doc.load_file(modelName);
	pugi::xml_node tree = doc.append_child("Para");
	tree.append_attribute("MINDR") = MINDR;
	tree.append_attribute("MAXFAR") = MAXFAR;
	tree.append_attribute("treeLevel") = tree_level;
	tree.append_attribute("minSamples") = minSamples;
	tree.append_attribute("negRatio") = negRatio;
	tree.append_attribute("minNegRatio") = minNegRatio;
	tree.append_attribute("numPosSample") = numPosSample;
	doc.save_file(modelName);
}

