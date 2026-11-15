#include "engine.h"
#include <iostream>
int main(){try{MotionEngine e;e.init();std::cout<<e.test()<<std::endl;return 0;}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
