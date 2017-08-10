CC = g++ -fopenmp
# 可执行文件
TARGET = test
# C文件
SRCS = pugixml.cpp Adaboost.cpp Dataset.cpp DQT.cpp main.cpp Parameter.cpp
# 目标文件
OBJS = $(SRCS:.cpp=.o)
# 库文件
	DLIBS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui
# 链接为可执行文件
$(TARGET):$(OBJS)
	$(CC) -o  $@ $^ $(DLIBS) -O3 -funroll-loops  
clean:
	rm -rf $(TARGET) $(OBJS)
# 编译规则 $@代表目标文件 $< 代表第一个依赖文件
%.o:%.cpp
	$(CC) -o $@ -c $< -Wall -g
