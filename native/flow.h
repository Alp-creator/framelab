#pragma once
#include <android/NeuralNetworks.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Coarse-to-fine block matching for the GPU pipeline.
// The NPU computes the squared-error cost volume of the finest level (240 blocks x 81
// candidates); the two small coarse levels and the patch packing run on the CPU.
// Still analytic block matching, not a trained interpolation network.

// One NNAPI graph: costs[n][c] = sum_k (candidates[n][c][k] - reference[n][k])^2
struct CostNet {
#ifndef FLOW_SUB
#define FLOW_SUB 2
#endif
    // Each 8x8 block is compared on every FLOW_SUB-th pixel: 4x less data per NPU call at SUB=2.
    static constexpr int C=81, SUB=FLOW_SUB, K=64/(SUB*SUB);
    const int N;
    // Small pyramid levels run on the CPU: an NPU round trip costs more than their whole computation.
    bool cpu=false;
    ANeuralNetworksModel* model=nullptr;
    ANeuralNetworksCompilation* compilation=nullptr;
    std::vector<float> reference, candidates, costs;
    std::string device, detail;
    double lastMs=0;
    explicit CostNet(int n):N(n),reference(size_t(n)*K),candidates(size_t(n)*C*K),costs(size_t(n)*C) {}
    CostNet(const CostNet&)=delete;
    ~CostNet(){if(compilation)ANeuralNetworksCompilation_free(compilation);if(model)ANeuralNetworksModel_free(model);}
    static void check(int s,const char* where){if(s)throw std::runtime_error(std::string(where)+": NNAPI status "+std::to_string(s));}
    void init(){
        if(cpu)return;
        check(ANeuralNetworksModel_create(&model),"model");
        uint32_t rd[]={1,uint32_t(N),1,K}, cd[]={1,uint32_t(N),C,K}, od[]={uint32_t(N*C),1}, wd[]={1,K}, bd[]={1};
        ANeuralNetworksOperandType rt{ANEURALNETWORKS_TENSOR_FLOAT32,4,rd,0,0},ct{ANEURALNETWORKS_TENSOR_FLOAT32,4,cd,0,0},ot{ANEURALNETWORKS_TENSOR_FLOAT32,2,od,0,0};
        ANeuralNetworksOperandType scalar{ANEURALNETWORKS_INT32,0,nullptr,0,0},weights{ANEURALNETWORKS_TENSOR_FLOAT32,2,wd,0,0},bias{ANEURALNETWORKS_TENSOR_FLOAT32,1,bd,0,0};
        const ANeuralNetworksOperandType* types[]={&rt,&ct,&scalar,&ct,&ct,&weights,&bias,&ot};
        for(auto t:types)check(ANeuralNetworksModel_addOperand(model,t),"operand");
        int32_t none=0; float ones[K];std::fill(ones,ones+K,1.f);float zero=0;
        check(ANeuralNetworksModel_setOperandValue(model,2,&none,4),"activation");
        check(ANeuralNetworksModel_setOperandValue(model,5,ones,sizeof(ones)),"weights");
        check(ANeuralNetworksModel_setOperandValue(model,6,&zero,4),"bias");
        uint32_t sub[]={1,0,2},sq[]={3,3,2},sum[]={4,5,6,2},o3=3,o4=4,o7=7,ins[]={0,1};
        check(ANeuralNetworksModel_addOperation(model,ANEURALNETWORKS_SUB,3,sub,1,&o3),"sub");
        check(ANeuralNetworksModel_addOperation(model,ANEURALNETWORKS_MUL,3,sq,1,&o4),"square");
        check(ANeuralNetworksModel_addOperation(model,ANEURALNETWORKS_FULLY_CONNECTED,4,sum,1,&o7),"reduce via FC");
        check(ANeuralNetworksModel_identifyInputsAndOutputs(model,2,ins,1,&o7),"io");
        check(ANeuralNetworksModel_finish(model),"finish model");
        uint32_t count=0; check(ANeuralNetworks_getDeviceCount(&count),"device count");
        std::ostringstream log;
        // MediaTek accelerators only, MDLA first. The NNAPI CPU reference device is never used.
        for(int pass=0;pass<2&&!compilation;pass++)for(uint32_t i=0;i<count;i++){
            ANeuralNetworksDevice* d=nullptr;const char* name="?";int32_t type=0;
            check(ANeuralNetworks_getDevice(i,&d),"device");check(ANeuralNetworksDevice_getName(d,&name),"name");check(ANeuralNetworksDevice_getType(d,&type),"type");
            std::string nm(name);
            if(type!=ANEURALNETWORKS_DEVICE_ACCELERATOR||nm.find("mtk")==std::string::npos)continue;
            if((nm.find("mdla")!=std::string::npos)!=(pass==0))continue;
            bool supported[3]={false,false,false};
            int s=ANeuralNetworksModel_getSupportedOperationsForDevices(model,&d,1,supported);
            log<<nm<<" support="<<supported[0]<<supported[1]<<supported[2]<<"; ";
            if(s||!supported[0]||!supported[1]||!supported[2])continue;
            ANeuralNetworksCompilation* comp=nullptr;
            s=ANeuralNetworksCompilation_createForDevices(model,&d,1,&comp);
            if(!s)s=ANeuralNetworksCompilation_setPreference(comp,ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
            if(!s)s=ANeuralNetworksCompilation_finish(comp);
            if(s){if(comp)ANeuralNetworksCompilation_free(comp);log<<"compile="<<s<<"; ";continue;}
            compilation=comp;device=nm;break;
        }
        detail=log.str();
        if(!compilation)throw std::runtime_error("NPU graph unavailable. "+detail);
    }
    void compute(){
        if(cpu){
            for(int n=0;n<N;n++){const float* r=&reference[size_t(n)*K];for(int c=0;c<C;c++){const float* q=&candidates[(size_t(n)*C+c)*K];float s=0;for(int k=0;k<K;k++){float d=q[k]-r[k];s+=d*d;}costs[size_t(n)*C+c]=s;}}
            lastMs=0;return;
        }
        ANeuralNetworksExecution* e=nullptr;
        check(ANeuralNetworksExecution_create(compilation,&e),"execution");
        auto start=std::chrono::steady_clock::now();
        int s=ANeuralNetworksExecution_setInput(e,0,nullptr,reference.data(),reference.size()*4);
        if(!s)s=ANeuralNetworksExecution_setInput(e,1,nullptr,candidates.data(),candidates.size()*4);
        if(!s)s=ANeuralNetworksExecution_setOutput(e,0,nullptr,costs.data(),costs.size()*4);
        if(!s)s=ANeuralNetworksExecution_compute(e);
        ANeuralNetworksExecution_free(e);
        check(s,"compute");
        lastMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
};

struct FlowEngine {
    // Finest level is 160x96 luma -> 20x12 blocks. Search reach: +-4 px per level,
    // doubled at each finer level => +-28 px at 160 wide (about 17% of the frame).
    static constexpr int LW=160, LH=96, B=8, R=4, FX=LW/B, FY=LH/B;
    static constexpr int SCALE=4; // flow byte = displacement*SCALE + 128
    CostNet l0{(LW/4/B)*(LH/4/B)}, l1{(LW/2/B)*(LH/2/B)}, l2{FX*FY};
    double npuMs=0; int unreliable=0;
    struct Field{int nx=0,ny=0;std::vector<int> x,y;};
    void init(){l0.cpu=l1.cpu=true;l0.init();l1.init();l2.init();}
    std::string info()const{return l2.device;}

    static void half(const std::vector<float>& in,int w,int h,std::vector<float>& out){
        out.resize((w/2)*(h/2));
        for(int y=0;y<h/2;y++)for(int x=0;x<w/2;x++)
            out[y*(w/2)+x]=.25f*(in[2*y*w+2*x]+in[2*y*w+2*x+1]+in[(2*y+1)*w+2*x]+in[(2*y+1)*w+2*x+1]);
    }
    // Backward motion: a block at p in `b` (newer) came from p+v in `a` (older).
    Field level(CostNet& net,const std::vector<float>& a,const std::vector<float>& b,int W,int H,const Field* parent,bool finest){
        const int nx=W/B,ny=H/B,C=CostNet::C,K=CostNet::K,S=CostNet::SUB;const float unit=K/64.f; // thresholds were tuned for 64 samples
        Field f;f.nx=nx;f.ny=ny;f.x.assign(nx*ny,0);f.y.assign(nx*ny,0);
        for(int by=0;by<ny;by++)for(int bx=0;bx<nx;bx++){
            int n=by*nx+bx,ox=0,oy=0;
            if(parent){int p=(by/2)*parent->nx+bx/2;ox=2*parent->x[p];oy=2*parent->y[p];f.x[n]=ox;f.y[n]=oy;}
            for(int yy=0;yy<B;yy+=S)for(int xx=0;xx<B;xx+=S){
                int x=bx*B+xx,y=by*B+yy,k=(yy/S)*(B/S)+xx/S;
                net.reference[n*K+k]=b[y*W+x];
                for(int sy=-R;sy<=R;sy++){
                    const float* row=&a[std::clamp(y+oy+sy,0,H-1)*W];
                    for(int sx=-R;sx<=R;sx++)net.candidates[(size_t(n)*C+(sy+R)*9+(sx+R))*K+k]=row[std::clamp(x+ox+sx,0,W-1)];
                }
            }
        }
        net.compute();npuMs+=net.lastMs;
        for(int n=0;n<nx*ny;n++){
            const float* cost=&net.costs[size_t(n)*C];
            int best=40;float score=cost[40];
            for(int c=0;c<C;c++){
                int sx=c%9-R,sy=c/9-R;float s=cost[c]+0.002f*unit*(sx*sx+sy*sy);
                if(s<score){score=s;best=c;}
            }
            if(cost[40]-score<0.01f*unit)best=40; // no clear winner: keep the inherited motion
            int vx=f.x[n]+best%9-R,vy=f.y[n]+best/9-R;
            if(finest){
                float raw=cost[best];
                if(vx||vy){
                    // HUD and other static content must not inherit camera motion.
                    int bx=n%nx,by=n/nx;float zero=0;
                    for(int yy=0;yy<B;yy+=S)for(int xx=0;xx<B;xx+=S){int i=(by*B+yy)*W+bx*B+xx;float d=a[i]-b[i];zero+=d*d;}
                    // A motion vector must also beat "no motion" clearly, otherwise it is a chance match.
                    if(zero<=raw*2.f+0.003f*unit){vx=vy=0;raw=zero;}
                }
                if(raw>2.5f*unit){vx=vy=0;unreliable++;} // no usable match: plain blend
            }
            f.x[n]=vx;f.y[n]=vy;
        }
        return f;
    }
    // Inputs: 160x96 RGBA with luma in the first channel. Output: 20x12 RGBA flow.
    // Returns false when the two frames look unrelated (scene cut).
    bool flow(const uint8_t* oldRgba,const uint8_t* nowRgba,uint8_t* out){
        std::vector<float> a2(LW*LH),b2(LW*LH),a1,b1,a0,b0;
        for(int i=0;i<LW*LH;i++){a2[i]=oldRgba[4*i]/255.f;b2[i]=nowRgba[4*i]/255.f;}
        half(a2,LW,LH,a1);half(b2,LW,LH,b1);half(a1,LW/2,LH/2,a0);half(b1,LW/2,LH/2,b0);
        npuMs=0;unreliable=0;
        Field f0=level(l0,a0,b0,LW/4,LH/4,nullptr,false);
        Field f1=level(l1,a1,b1,LW/2,LH/2,&f0,false);
        Field f2=level(l2,a2,b2,LW,LH,&f1,true);
        for(int n=0;n<FX*FY;n++){
            out[4*n]=uint8_t(std::clamp(f2.x[n]*SCALE+128,0,255));
            out[4*n+1]=uint8_t(std::clamp(f2.y[n]*SCALE+128,0,255));
            out[4*n+2]=0;out[4*n+3]=255;
        }
        return unreliable*5<FX*FY*2; // cut when more than 40% of blocks have no match
    }
};
