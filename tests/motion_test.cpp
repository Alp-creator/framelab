#include "../native/engine.h"
#include <fstream>
#include <iostream>
static uint32_t texture(int x,int y){
    auto channel=[](float x){return std::clamp(int(x),0,255);};
    int r=channel(128+65*std::sin(x*.17)+50*std::cos(y*.19));
    int g=channel(128+65*std::sin(x*.11+y*.09)+50*std::cos(y*.27));
    int b=channel(128+65*std::cos(x*.23)+50*std::sin(y*.13));
    return 0xff000000|(r<<16)|(g<<8)|b;
}
static void ppm(const char* path,const std::vector<uint32_t>& p,int w,int h){std::ofstream f(path,std::ios::binary);f<<"P6\n"<<w<<" "<<h<<"\n255\n";for(auto v:p){char c[3]={char(v>>16),char(v>>8),char(v)};f.write(c,3);}}
int main(){try{
    MotionEngine engine;engine.init();std::cout<<engine.test()<<"\n";
    constexpr int w=480,h=288;std::vector<uint32_t>a(w*h),b(w*h),expected(w*h),out(w*h);
    for(int y=0;y<h;y++)for(int x=0;x<w;x++){a[y*w+x]=texture(x,y);b[y*w+x]=texture(x-6,y);expected[y*w+x]=texture(x-3,y);}
    auto start=std::chrono::steady_clock::now();engine.interpolate(a.data(),b.data(),out.data(),w,h);
    double total=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    double mse=0,blend=0;int count=0;
    for(int y=32;y<h-32;y++)for(int x=32;x<w-32;x++)for(int s=0;s<=16;s+=8){int i=y*w+x;float gt=(expected[i]>>s)&255;float got=(out[i]>>s)&255;float simple=(((a[i]>>s)&255)+((b[i]>>s)&255))*.5f;mse+=(got-gt)*(got-gt);blend+=(simple-gt)*(simple-gt);count++;}
    mse/=count;blend/=count;std::cout<<"translation midpoint MSE="<<mse<<"\nplain blend MSE="<<blend<<"\nwhole interpolation ms="<<total<<"\n";
    if(mse>=blend*.5)throw std::runtime_error("Motion midpoint did not improve over crossfade");
    ppm("/data/local/tmp/framelab-midpoint.ppm",out,w,h);ppm("/data/local/tmp/framelab-ground-truth.ppm",expected,w,h);
    engine.interpolate(a.data(),a.data(),out.data(),w,h);
    int maxError=0;for(size_t i=0;i<a.size();i++)for(int s=0;s<=16;s+=8)maxError=std::max(maxError,std::abs(int((a[i]>>s)&255)-int((out[i]>>s)&255)));
    std::cout<<"static max channel error="<<maxError<<"\n";if(maxError>1)throw std::runtime_error("Static frame changed");
    std::cout<<"PASS: accelerator numerics, translation midpoint, static frame\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
