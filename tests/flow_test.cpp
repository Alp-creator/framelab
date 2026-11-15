#include "../native/flow.h"
#include <iostream>
// Runs on the tablet: checks that the NPU pyramid recovers known motions.
// Non-periodic value noise at three scales, like real image content.
static float lattice(int x,int y,uint32_t seed){uint32_t h=uint32_t(x)*374761393u+uint32_t(y)*668265263u+seed*2246822519u;h=(h^(h>>13))*1274126177u;return float((h^(h>>16))&0xffff)/65535.f;}
static float noise(float x,float y,float cell,uint32_t seed){
    x=x/cell+1000;y=y/cell+1000;int ix=int(std::floor(x)),iy=int(std::floor(y));float fx=x-ix,fy=y-iy;fx=fx*fx*(3-2*fx);fy=fy*fy*(3-2*fy);
    return (lattice(ix,iy,seed)*(1-fx)+lattice(ix+1,iy,seed)*fx)*(1-fy)+(lattice(ix,iy+1,seed)*(1-fx)+lattice(ix+1,iy+1,seed)*fx)*fy;
}
static float tex(float x,float y){return .45f*noise(x,y,23,1)+.35f*noise(x,y,9,2)+.2f*noise(x,y,4,3);}
static void fill(std::vector<uint8_t>& p,float ox,float oy,bool hud){
    const int W=FlowEngine::LW,H=FlowEngine::LH;p.assign(W*H*4,255);
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
        float v=tex(x-ox,y-oy);
        if(hud&&x<32&&y<16)v=((x/4+y/4)&1)?.9f:.1f; // static overlay in the corner
        p[4*(y*W+x)]=uint8_t(std::clamp(v,0.f,1.f)*255);
    }
}
int main(){try{
    FlowEngine e;e.init();std::cout<<"device="<<e.info()<<"\n";
    const int FX=FlowEngine::FX,FY=FlowEngine::FY;std::vector<uint8_t> a,b,out(FX*FY*4);
    struct Case{int dx,dy;}cases[]={{0,0},{3,0},{-6,2},{12,-5},{-18,4},{22,9}};
    for(auto c:cases){
        fill(a,0,0,true);fill(b,c.dx,c.dy,true);
        // Content moved by +d from old to new, so backward flow is -d.
        bool ok=e.flow(a.data(),b.data(),out.data());int good=0,total=0,hud=0;
        for(int by=2;by<FY-2;by++)for(int bx=5;bx<FX-4;bx++){int n=by*FX+bx;total++;
            int vx=(out[4*n]-128)/FlowEngine::SCALE,vy=(out[4*n+1]-128)/FlowEngine::SCALE;if(std::abs(vx+c.dx)<=1&&std::abs(vy+c.dy)<=1)good++;} // low-contrast blocks may be one step off
        for(int by=0;by<2;by++)for(int bx=0;bx<4;bx++){int n=by*FX+bx;if(out[4*n]==128&&out[4*n+1]==128)hud++;}
        {int n=6*FX+10;std::cout<<"  centre block flow=("<<(out[4*n]-128)/FlowEngine::SCALE<<","<<(out[4*n+1]-128)/FlowEngine::SCALE<<") ";}
        std::cout<<"shift("<<c.dx<<","<<c.dy<<") exact="<<good<<"/"<<total<<" static-hud="<<hud<<"/8 cut="<<!ok<<" npu-ms="<<e.npuMs<<"\n";
        // Large shifts reveal new content at the borders that has no match by definition.
        bool large=std::abs(c.dx)>12||std::abs(c.dy)>12;
        if(!ok||good*10<total*(large?6:9)||hud<8)throw std::runtime_error("flow mismatch");
    }
    fill(a,0,0,false);for(size_t i=0;i<b.size();i+=4)b[i]=uint8_t((i*2654435761u)>>24);
    if(e.flow(a.data(),b.data(),out.data()))throw std::runtime_error("scene cut not detected");
    std::cout<<"scene cut detected\nPASS\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
