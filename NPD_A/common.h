//
//  common.h
//  NPD_A
//
//  Created by 陈石涛 on 29/06/2017.
//  Copyright © 2017 Shitao Chen. All rights reserved.
//

#ifndef common_h
#include<iostream>
#include<string>
#include<fstream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "Parameter.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#define common_h
typedef unsigned char uchar;

void IndexSort(vector<int>& _index, vector<double>& _content, int direction = 1);
#endif /* common_h */
