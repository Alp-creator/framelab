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

// Analytic block matching, not a trained neural frame-interpolation model.
// NPU computes the squared-error cost volume. CPU packs patches, selects
// displacements, and resamples pixels. No NNAPI CPU fallback is requested.
struct MotionEngine {
    static constexpr int LW=160, LH=96, B=8, R=4, C=81, NX=20, NY=12, N=240;
    ANeuralNetworksModel* model=nullptr;
    ANeuralNetworksCompilation* compilation=nullptr;
    std::vector<float> reference, candidates, costs;
    std::string device, detail;
    double lastMs=0;
    MotionEngine():reference(N*64),candidates(N*C*64),costs(N*C) {}
    ~MotionEngine(){if(compilation) ANeuralNetworksCompilation_free(compilation);if(model) ANeuralNetworksModel_free(model);}
    static void check(int s,const char* where){if(s)throw std::runtime_error(std::string(where)+": NNAPI status "+std::to_string(s));}
    void init(){
        check(ANeuralNetworksModel_create(&model),"model");
        uint32_t rd[]={1,N,1,64}, cd[]={1,N,C,64}, od[]={N*C,1}, wd[]={1,64}, bd[]={1};
        ANeuralNetworksOperandType rt{ANEURALNETWORKS_TENSOR_FLOAT32,4,rd,0,0},ct{ANEURALNETWORKS_TENSOR_FLOAT32,4,cd,0,0},ot{ANEURALNETWORKS_TENSOR_FLOAT32,2,od,0,0};
        ANeuralNetworksOperandType scalar{ANEURALNETWORKS_INT32,0,nullptr,0,0},weights{ANEURALNETWORKS_TENSOR_FLOAT32,2,wd,0,0},bias{ANEURALNETWORKS_TENSOR_FLOAT32,1,bd,0,0};
        const ANeuralNetworksOperandType* types[]={&rt,&ct,&scalar,&ct,&ct,&weights,&bias,&ot};
        for(auto t:types)check(ANeuralNetworksModel_addOperand(model,t),"operand");
        int32_t none=0; float ones[64];std::fill(ones,ones+64,1.f);float zero=0;
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
        // Prefer the MediaTek neural accelerator over its DSP shim.
        for(int pass=0;pass<2&&!compilation;pass++)for(uint32_t i=0;i<count;i++){
            ANeuralNetworksDevice* d=nullptr;const char* name="?";int32_t type=0;
            check(ANeuralNetworks_getDevice(i,&d),"device");check(ANeuralNetworksDevice_getName(d,&name),"name");check(ANeuralNetworksDevice_getType(d,&type),"type");
            std::string nm(name);
            if(type!=ANEURALNETWORKS_DEVICE_ACCELERATOR || nm.find("mtk")==std::string::npos)continue;
            bool preferred=nm.find("mdla")!=std::string::npos;
            if(preferred!=(pass==0))continue;
            bool supported[3]={false,false,false};
            int s=ANeuralNetworksModel_getSupportedOperationsForDevices(model,&d,1,supported);
            log<<nm<<" type="<<type<<" support="<<supported[0]<<supported[1]<<supported[2]<<" status="<<s<<"; ";
            if(s||!supported[0]||!supported[1]||!supported[2])continue;
            ANeuralNetworksCompilation* comp=nullptr;
            s=ANeuralNetworksCompilation_createForDevices(model,&d,1,&comp);
            if(!s)s=ANeuralNetworksCompilation_setPreference(comp,ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
            if(!s)s=ANeuralNetworksCompilation_finish(comp);
            if(s){if(comp)ANeuralNetworksCompilation_free(comp);log<<"compile="<<s<<"; ";continue;}
            compilation=comp;device=nm;break;
        }
        detail=log.str();
        if(!compilation)throw std::runtime_error("NPU graph unavailable. "+detail);
    }
    void compute(){
        if(!compilation)throw std::runtime_error("NPU not initialized");
        ANeuralNetworksExecution* e=nullptr;
        check(ANeuralNetworksExecution_create(compilation,&e),"execution");
        auto start=std::chrono::steady_clock::now();
        try{
            check(ANeuralNetworksExecution_setInput(e,0,nullptr,reference.data(),reference.size()*4),"input0");
            check(ANeuralNetworksExecution_setInput(e,1,nullptr,candidates.data(),candidates.size()*4),"input1");
            check(ANeuralNetworksExecution_setOutput(e,0,nullptr,costs.data(),costs.size()*4),"output");
            check(ANeuralNetworksExecution_compute(e),"compute");
        }catch(...){ANeuralNetworksExecution_free(e);throw;}
        lastMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        ANeuralNetworksExecution_free(e);
    }
    std::string test(){
        for(size_t i=0;i<reference.size();i++)reference[i]=float((i*7)%251)/251;
        for(size_t i=0;i<candidates.size();i++)candidates[i]=float((i*11)%253)/253;
        std::vector<double> ms;
        for(int t=0;t<6;t++){compute();if(t)ms.push_back(lastMs);}
        float maxErr=0;
        for(int n=0;n<N;n++)for(int c=0;c<C;c++){
            float expected=0;for(int k=0;k<64;k++){float v=candidates[(n*C+c)*64+k]-reference[n*64+k];expected+=v*v;}
            if(!std::isfinite(costs[n*C+c]))throw std::runtime_error("Nonfinite NPU output");
            maxErr=std::max(maxErr,std::abs(costs[n*C+c]-expected));
        }
        if(maxErr>0.15f)throw std::runtime_error("NPU numeric test failed: "+std::to_string(maxErr));
        std::sort(ms.begin(),ms.end());
        std::ostringstream s;s<<"device="<<device<<"\nCPU fallback=disabled\ncost entries="<<N*C<<"\nmedian NPU call ms="<<ms[ms.size()/2]<<"\nmax absolute error="<<maxErr<<"\n"<<detail;
        return s.str();
    }
    std::vector<float> luma(const uint32_t* rgba,int w,int h){
        std::vector<float> out(LW*LH);
        for(int y=0;y<LH;y++)for(int x=0;x<LW;x++){
            int sx=std::min(w-1,(2*x+1)*w/(2*LW)),sy=std::min(h-1,(2*y+1)*h/(2*LH));
            uint32_t p=rgba[sy*w+sx]; // Bitmap int[] uses ARGB, independent of host byte order.
            out[y*LW+x]=(0.299f*((p>>16)&255)+0.587f*((p>>8)&255)+0.114f*(p&255))/255.f;
        }return out;
    }
    // Backward flow: for each block in the newer frame find its position in old.
    void motion(const uint32_t* old,const uint32_t* now,int w,int h,std::vector<float>& dx,std::vector<float>& dy){
        auto a=luma(old,w,h),b=luma(now,w,h);
        for(int by=0;by<NY;by++)for(int bx=0;bx<NX;bx++){
            int n=by*NX+bx;
            for(int yy=0;yy<B;yy++)for(int xx=0;xx<B;xx++){
                int x=bx*B+xx,y=by*B+yy,k=yy*B+xx;
                reference[n*64+k]=b[y*LW+x];
                for(int sy=-R;sy<=R;sy++)for(int sx=-R;sx<=R;sx++){
                    int c=(sy+R)*9+(sx+R);
                    candidates[(n*C+c)*64+k]=a[std::clamp(y+sy,0,LH-1)*LW+std::clamp(x+sx,0,LW-1)];
                }
            }
        }
        compute();dx.resize(N);dy.resize(N);
        for(int n=0;n<N;n++){
            int best=40;float score=costs[n*C+40];
            for(int c=0;c<C;c++){
                int sx=c%9-R,sy=c/9-R;
                float s=costs[n*C+c]+0.002f*(sx*sx+sy*sy);
                if(s<score){score=s;best=c;}
            }
            // Suppress unreliable matches, scene cuts and nearly static blocks.
            if(score>2.5f||costs[n*C+40]-score<0.01f)best=40;
            dx[n]=float(best%9-R)*w/LW;dy[n]=float(best/9-R)*h/LH;
        }
    }
    static uint32_t sample(const uint32_t* p,int w,int h,float x,float y){
        x=std::clamp(x,0.f,float(w-1));y=std::clamp(y,0.f,float(h-1));
        int ix=int(x),iy=int(y),jx=std::min(ix+1,w-1),jy=std::min(iy+1,h-1);float fx=x-ix,fy=y-iy;uint32_t v=0xff000000;
        for(int s=0;s<=16;s+=8){float a=((p[iy*w+ix]>>s)&255)*(1-fx)+((p[iy*w+jx]>>s)&255)*fx;float b=((p[jy*w+ix]>>s)&255)*(1-fx)+((p[jy*w+jx]>>s)&255)*fx;v|=uint32_t(a*(1-fy)+b*fy+0.5f)<<s;}return v;
    }
    void interpolate(const uint32_t* old,const uint32_t* now,uint32_t* out,int w,int h){
        std::vector<float> dx,dy;motion(old,now,w,h,dx,dy);
        for(int y=0;y<h;y++)for(int x=0;x<w;x++){
            float gx=std::clamp((x+.5f)*NX/w-.5f,0.f,float(NX-1)),gy=std::clamp((y+.5f)*NY/h-.5f,0.f,float(NY-1));
            int ix=int(gx),iy=int(gy),jx=std::min(ix+1,NX-1),jy=std::min(iy+1,NY-1);float fx=gx-ix,fy=gy-iy;
            auto interp=[&](const std::vector<float>& v){return (v[iy*NX+ix]*(1-fx)+v[iy*NX+jx]*fx)*(1-fy)+(v[jy*NX+ix]*(1-fx)+v[jy*NX+jx]*fx)*fy;};
            float vx=interp(dx)*.5f,vy=interp(dy)*.5f;
            uint32_t a=sample(old,w,h,x+vx,y+vy),b=sample(now,w,h,x-vx,y-vy);
            out[y*w+x]=0xff000000|(((a&0xfefefe)+(b&0xfefefe))>>1);
        }
    }
};
