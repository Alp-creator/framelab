#pragma once
#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "sr_model.h"

// x2 AI super resolution on the NPU (NNAPI). A small trained CNN (tools/train_sr.py).
//
//   in : low-res luma, one byte per pixel, [1,H,W,1] uint8 (value/255)
//   out: four bytes per low-res pixel, [1,H,W,4] uint8. Byte 2*dy+dx is the detail of high-res pixel
//        (2y+dy, 2x+dx) that bilinear upscaling misses, stored as 128 + 255*detail.
//
// The GPU adds the detail to its own bilinear upscale of the colour frame, so colour never leaves the GPU
// and only 1 byte per low-res pixel travels to the NPU.
namespace sr {

enum Kind { Q8_CHANNEL, Q8_TENSOR, F16, F32_RELAXED, KINDS };
inline const char* kindName(int k){static const char* n[]={"int8 per-channel","int8 per-tensor","float16","float32 relaxed"};return k>=0&&k<KINDS?n[k]:"?";}
inline void check(int s,const char* where){if(s!=ANEURALNETWORKS_NO_ERROR)throw std::runtime_error(std::string(where)+": NNAPI status "+std::to_string(s));}
inline double nowMs(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}

struct Device{ANeuralNetworksDevice* d;std::string name;};
// Neural accelerators only: never the NNAPI CPU reference, never the GPU the game needs.
// MediaTek order: MDLA (the NPU itself), then the Neuron runtime, then DSP/VPU.
inline std::vector<Device> accelerators(){
    uint32_t n=0;check(ANeuralNetworks_getDeviceCount(&n),"device count");
    std::vector<Device> out;
    for(uint32_t i=0;i<n;i++){
        ANeuralNetworksDevice* d=nullptr;const char* name="?";int32_t type=0;
        check(ANeuralNetworks_getDevice(i,&d),"device");check(ANeuralNetworksDevice_getName(d,&name),"name");check(ANeuralNetworksDevice_getType(d,&type),"type");
        if(type==ANEURALNETWORKS_DEVICE_ACCELERATOR)out.push_back({d,name});
    }
    auto rank=[](const std::string& s){return s.find("mdla")!=std::string::npos?0:s.find("neuron")!=std::string::npos?1:2;};
    std::stable_sort(out.begin(),out.end(),[&](const Device& a,const Device& b){return rank(a.name)<rank(b.name);});
    return out;
}

// The exact numbers one graph variant uses; the CPU reference below uses the same ones.
struct Layer{
    int cin,cout;bool relu;float inScale,outScale;int outZero;
    std::vector<int8_t> qw;std::vector<float> qs;std::vector<int32_t> qb;   // int8 per-channel
    std::vector<uint8_t> uw;float us=0;std::vector<int32_t> ub;             // int8 per-tensor (zero point 128)
    std::vector<float> fw,fb;                                               // float
    std::vector<double> mult;                                               // requantization per channel
};
inline std::vector<Layer> layers(const srmodel::Model& m,Kind kind){
    std::vector<Layer> out;float in=1.f/255;
    for(int l=0;l<m.layers;l++){
        const srmodel::Layer& s=m.layer[l];Layer L;L.cin=s.cin;L.cout=s.cout;L.relu=s.relu;L.inScale=in;L.outScale=s.outScale;L.outZero=s.outZero;
        const int k=9*s.cin,n=s.cout*k;
        L.fw.assign(s.w,s.w+n);L.fb.assign(s.b,s.b+s.cout);
        L.qw.assign(s.qw,s.qw+n);L.qs.assign(s.qs,s.qs+s.cout);L.qb.assign(s.qb,s.qb+s.cout);
        float mx=1e-8f;for(int i=0;i<n;i++)mx=std::max(mx,std::fabs(s.w[i]));L.us=mx/127.f;
        L.uw.resize(n);for(int i=0;i<n;i++)L.uw[i]=uint8_t(std::clamp<long>(std::lround(s.w[i]/L.us)+128,0,255));
        L.ub.resize(s.cout);L.mult.resize(s.cout);
        for(int c=0;c<s.cout;c++){
            L.ub[c]=int32_t(std::lround(double(s.b[c])/(double(in)*L.us)));
            L.mult[c]=kind==Q8_TENSOR?double(in)*L.us/s.outScale:double(in)*s.qs[c]/s.outScale;
        }
        out.push_back(std::move(L));in=s.outScale;
    }
    return out;
}

// CPU reference of one output pixel (all four phases): exact integer arithmetic for the int8 variants,
// float for the others. Zero padding at every layer, as NNAPI's SAME padding does.
inline void reference(const std::vector<Layer>& L,Kind kind,const uint8_t* img,int W,int H,int x,int y,int out[4]){
    const int n=int(L.size());int r=n;std::vector<double> cur((2*r+1)*(2*r+1));
    for(int dy=-r;dy<=r;dy++)for(int dx=-r;dx<=r;dx++){int px=x+dx,py=y+dy;cur[(dy+r)*(2*r+1)+dx+r]=(px<0||py<0||px>=W||py>=H)?0:img[py*W+px]*(kind<=Q8_TENSOR?1.0:1.0/255);}
    for(int l=0;l<n;l++){
        const Layer& a=L[l];int ro=r-1,wi=2*r+1,wo=2*ro+1;std::vector<double> next(size_t(wo)*wo*a.cout,0.0);
        for(int dy=-ro;dy<=ro;dy++)for(int dx=-ro;dx<=ro;dx++){
            int px=x+dx,py=y+dy;double* o=&next[(size_t(dy+ro)*wo+dx+ro)*a.cout];
            if(px<0||py<0||px>=W||py>=H)continue; // outside the image every layer holds real zero
            for(int c=0;c<a.cout;c++){
                double acc=kind==Q8_CHANNEL?a.qb[c]:kind==Q8_TENSOR?a.ub[c]:a.fb[c];
                for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
                    int qx=dx+kx-1,qy=dy+ky-1;if(px+kx-1<0||py+ky-1<0||px+kx-1>=W||py+ky-1>=H)continue;
                    const double* v=&cur[(size_t(qy+r)*wi+qx+r)*(l?L[l-1].cout:1)];
                    for(int i=0;i<a.cin;i++){
                        int wk=((c*3+ky)*3+kx)*a.cin+i;
                        double w=kind==Q8_CHANNEL?a.qw[wk]:kind==Q8_TENSOR?double(a.uw[wk])-128:a.fw[wk];
                        acc+=v[i]*w;
                    }
                }
                if(kind<=Q8_TENSOR){double q=double(std::lround(acc*a.mult[c]))+a.outZero;o[c]=std::clamp(q,0.0,255.0);}
                else if(l<n-1)o[c]=std::max(acc,0.0);
                else o[c]=std::clamp(double(std::lround(acc/a.outScale))+a.outZero,0.0,255.0);
            }
        }
        cur.swap(next);r=ro;
    }
    for(int c=0;c<4;c++)out[c]=int(cur[c]);
}

// Deterministic test image: gradients, rings, hard edges, thin lines, text-like strokes and noise.
inline void testPattern(uint8_t* p,int W,int H){
    uint32_t s=12345;
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
        s=s*1664525u+1013904223u;
        double v=128+55*std::sin(x*0.071)*std::cos(y*0.053)+40*std::sin((x*x+y*y)*0.0009);
        if(((x/37)+(y/23))%3==0)v+=45;
        if(x%29==0||y%31==0)v=255-v;
        if((x/5+y/7)%11==0&&(y%13)<9)v=20;
        v+=int(s>>27)-16;
        p[size_t(y)*W+x]=uint8_t(std::clamp(v,0.0,255.0));
    }
}

class Net{
public:
    const srmodel::Model& model;const Kind kind;const int W,H;const Device dev;
    std::vector<Layer> L;std::string log;
    bool supported=false;double compileMs=0,lastMs=0,hwMs=-1,driverMs=-1;
    size_t inBytes,outBytes;uint8_t* in=nullptr;uint8_t* out=nullptr;

    Net(const srmodel::Model& m,Kind k,int w,int h,const Device& d,const std::string& cacheDir="")
        :model(m),kind(k),W(w),H(h),dev(d),L(layers(m,k)),inBytes(size_t(w)*h),outBytes(size_t(w)*h*4){
        try{build(cacheDir);}catch(...){release();throw;}
    }
    ~Net(){release();}
    Net(const Net&)=delete;

    // One NPU pass over `in`, result in `out`. Returns wall milliseconds.
    double run(){
        double t=nowMs();
        int s=burst?ANeuralNetworksExecution_burstCompute(exec,burst):ANeuralNetworksExecution_compute(exec);
        if(s!=ANEURALNETWORKS_NO_ERROR&&burst){ANeuralNetworksBurst_free(burst);burst=nullptr;s=ANeuralNetworksExecution_compute(exec);}
        check(s,"compute");
        lastMs=nowMs()-t;
        uint64_t ns=UINT64_MAX;
        hwMs=ANeuralNetworksExecution_getDuration(exec,ANEURALNETWORKS_DURATION_ON_HARDWARE,&ns)==0&&ns!=UINT64_MAX?ns/1e6:-1;
        driverMs=ANeuralNetworksExecution_getDuration(exec,ANEURALNETWORKS_DURATION_IN_DRIVER,&ns)==0&&ns!=UINT64_MAX?ns/1e6:-1;
        return lastMs;
    }
    // Compares the NPU against the CPU reference on sampled pixels (borders included).
    // Returns the share of sampled values more than 3 steps off.
    double verify(double* meanErr,int* maxErr,int samples=800){
        testPattern(in,W,H);run();
        uint32_t s=777;double sum=0;int worst=0,bad=0,count=0;
        for(int i=0;i<samples;i++){
            s=s*1664525u+1013904223u;int x=int((s>>8)%uint32_t(W));s=s*1664525u+1013904223u;int y=int((s>>8)%uint32_t(H));
            if(i%10==0){x=i%20<10?0:W-1;} if(i%10==1){y=i%20<10?0:H-1;}
            int ref[4];reference(L,kind,in,W,H,x,y,ref);
            for(int c=0;c<4;c++){int e=std::abs(int(out[(size_t(y)*W+x)*4+c])-ref[c]);sum+=e;worst=std::max(worst,e);bad+=e>3;count++;}
        }
        *meanErr=sum/count;*maxErr=worst;return double(bad)/count;
    }
    std::string describe()const{
        std::ostringstream s;s<<dev.name<<" · "<<kindName(kind)<<" · "<<model.name<<" · "<<W<<"x"<<H<<" -> "<<2*W<<"x"<<2*H;return s.str();
    }
    int operations()const{return ops;}

private:
    ANeuralNetworksModel* nn=nullptr;ANeuralNetworksCompilation* comp=nullptr;ANeuralNetworksExecution* exec=nullptr;ANeuralNetworksBurst* burst=nullptr;
    ANeuralNetworksMemory* inMem=nullptr;ANeuralNetworksMemory* outMem=nullptr;int inFd=-1,outFd=-1;
    uint32_t operandCount=0;int ops=0;
    std::vector<std::shared_ptr<void>> keep; // constant data must outlive the model

    uint32_t operand(int32_t type,std::vector<uint32_t> dims,float scale=0,int32_t zero=0){
        ANeuralNetworksOperandType t{type,uint32_t(dims.size()),dims.empty()?nullptr:dims.data(),scale,zero};
        check(ANeuralNetworksModel_addOperand(nn,&t),"operand");return operandCount++;
    }
    uint32_t scalar(int32_t v){uint32_t k=operand(ANEURALNETWORKS_INT32,{});check(ANeuralNetworksModel_setOperandValue(nn,k,&v,4),"scalar");return k;}
    template<class T> uint32_t constant(int32_t type,std::vector<uint32_t> dims,const std::vector<T>& data,float scale=0,int32_t zero=0){
        uint32_t k=operand(type,dims,scale,zero);auto copy=std::make_shared<std::vector<T>>(data);keep.push_back(copy);
        check(ANeuralNetworksModel_setOperandValue(nn,k,copy->data(),copy->size()*sizeof(T)),"constant");return k;
    }
    void op(int32_t type,std::vector<uint32_t> ins,uint32_t outIndex){check(ANeuralNetworksModel_addOperation(nn,type,uint32_t(ins.size()),ins.data(),1,&outIndex),"operation");ops++;}

    void build(const std::string& cacheDir){
        const uint32_t w=uint32_t(W),h=uint32_t(H);
        check(ANeuralNetworksModel_create(&nn),"model");
        uint32_t input=operand(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,{1,h,w,1},1.f/255,0);
        uint32_t pad=scalar(ANEURALNETWORKS_PADDING_SAME),one=scalar(1),relu=scalar(ANEURALNETWORKS_FUSED_RELU),none=scalar(ANEURALNETWORKS_FUSED_NONE);
        const bool q8=kind==Q8_CHANNEL||kind==Q8_TENSOR;
        const int32_t ft=kind==F16?ANEURALNETWORKS_TENSOR_FLOAT16:ANEURALNETWORKS_TENSOR_FLOAT32;
        uint32_t x=input;
        if(!q8){uint32_t f=operand(ft,{1,h,w,1});op(ANEURALNETWORKS_DEQUANTIZE,{input},f);x=f;}
        for(size_t l=0;l<L.size();l++){
            const Layer& a=L[l];const uint32_t co=uint32_t(a.cout),ci=uint32_t(a.cin);
            uint32_t filter,bias,y;
            if(kind==Q8_CHANNEL){
                filter=constant(ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL,{co,3,3,ci},a.qw);
                auto scales=std::make_shared<std::vector<float>>(a.qs);keep.push_back(scales);
                ANeuralNetworksSymmPerChannelQuantParams p{0,uint32_t(scales->size()),scales->data()};
                check(ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(nn,filter,&p),"per-channel scales");
                bias=constant(ANEURALNETWORKS_TENSOR_INT32,{co},a.qb,0.f,0);
            }else if(kind==Q8_TENSOR){
                filter=constant(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,{co,3,3,ci},a.uw,a.us,128);
                bias=constant(ANEURALNETWORKS_TENSOR_INT32,{co},a.ub,a.inScale*a.us,0);
            }else if(kind==F16){
                std::vector<__fp16> fw(a.fw.begin(),a.fw.end()),fb(a.fb.begin(),a.fb.end());
                filter=constant(ft,{co,3,3,ci},fw);bias=constant(ft,{co},fb);
            }else{filter=constant(ft,{co,3,3,ci},a.fw);bias=constant(ft,{co},a.fb);}
            y=q8?operand(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,{1,h,w,co},a.outScale,a.outZero):operand(ft,{1,h,w,co});
            op(ANEURALNETWORKS_CONV_2D,{x,filter,bias,pad,one,one,a.relu?relu:none},y);
            x=y;
        }
        if(!q8){uint32_t o=operand(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,{1,h,w,4},1.f/255,128);op(ANEURALNETWORKS_QUANTIZE,{x},o);x=o;}
        check(ANeuralNetworksModel_identifyInputsAndOutputs(nn,1,&input,1,&x),"io");
        if(kind==F32_RELAXED)check(ANeuralNetworksModel_relaxComputationFloat32toFloat16(nn,true),"relax");
        check(ANeuralNetworksModel_finish(nn),"finish model");

        std::unique_ptr<bool[]> support(new bool[ops]());
        ANeuralNetworksDevice* d=dev.d;
        check(ANeuralNetworksModel_getSupportedOperationsForDevices(nn,&d,1,support.get()),"supported ops");
        int yes=0;for(int i=0;i<ops;i++)yes+=support[i];
        supported=yes==ops;
        {std::ostringstream s;s<<"ops "<<yes<<"/"<<ops;log=s.str();}
        if(!supported)return;

        double t=nowMs();
        check(ANeuralNetworksCompilation_createForDevices(nn,&d,1,&comp),"compilation");
        check(ANeuralNetworksCompilation_setPreference(comp,ANEURALNETWORKS_PREFER_SUSTAINED_SPEED),"preference");
        ANeuralNetworksCompilation_setPriority(comp,ANEURALNETWORKS_PRIORITY_HIGH);
        if(!cacheDir.empty()){uint8_t token[ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN];cacheToken(token);ANeuralNetworksCompilation_setCaching(comp,cacheDir.c_str(),token);}
        check(ANeuralNetworksCompilation_finish(comp),"compile");
        compileMs=nowMs()-t;

        // Shared memory both sides map: the GPU readback lands directly in the NPU's input.
        inFd=ASharedMemory_create("framelab-sr-in",inBytes);outFd=ASharedMemory_create("framelab-sr-out",outBytes);
        if(inFd<0||outFd<0)throw std::runtime_error("shared memory");
        void* pi=mmap(nullptr,inBytes,PROT_READ|PROT_WRITE,MAP_SHARED,inFd,0);void* po=mmap(nullptr,outBytes,PROT_READ|PROT_WRITE,MAP_SHARED,outFd,0);
        if(pi==MAP_FAILED||po==MAP_FAILED){if(pi!=MAP_FAILED)munmap(pi,inBytes);if(po!=MAP_FAILED)munmap(po,outBytes);throw std::runtime_error("mmap");}
        in=static_cast<uint8_t*>(pi);out=static_cast<uint8_t*>(po);
        check(ANeuralNetworksExecution_create(comp,&exec),"execution");
        ANeuralNetworksExecution_setMeasureTiming(exec,true);
        check(ANeuralNetworksExecution_setReusable(exec,true),"reusable");
        bool mem=ANeuralNetworksMemory_createFromFd(inBytes,PROT_READ|PROT_WRITE,inFd,0,&inMem)==0&&ANeuralNetworksMemory_createFromFd(outBytes,PROT_READ|PROT_WRITE,outFd,0,&outMem)==0;
        if(mem)mem=ANeuralNetworksExecution_setInputFromMemory(exec,0,nullptr,inMem,0,inBytes)==0&&ANeuralNetworksExecution_setOutputFromMemory(exec,0,nullptr,outMem,0,outBytes)==0;
        if(!mem){ // driver without shared-memory support: plain pointers into the same mapping
            check(ANeuralNetworksExecution_setInput(exec,0,nullptr,in,inBytes),"input");
            check(ANeuralNetworksExecution_setOutput(exec,0,nullptr,out,outBytes),"output");
            log+=", pointer io";
        }
        if(ANeuralNetworksBurst_create(comp,&burst)!=0)burst=nullptr;
    }
    void cacheToken(uint8_t* t)const{
        std::ostringstream s;s<<"framelab-sr-2|"<<model.name<<"|"<<model.info<<"|"<<kind<<"|"<<W<<"x"<<H<<"|"<<dev.name;
        for(const Layer& a:L){for(float v:a.fw)s<<v<<',';s<<a.outScale;}
        std::string k=s.str();
        for(int part=0;part<ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN/8;part++){
            uint64_t h=1469598103934665603ull^uint64_t(part)*0x9e3779b97f4a7c15ull;
            for(unsigned char c:k){h^=c;h*=1099511628211ull;}
            std::memcpy(t+8*part,&h,8);
        }
    }
    void release(){
        if(burst)ANeuralNetworksBurst_free(burst);burst=nullptr;
        if(exec)ANeuralNetworksExecution_free(exec);exec=nullptr;
        if(inMem)ANeuralNetworksMemory_free(inMem);if(outMem)ANeuralNetworksMemory_free(outMem);inMem=outMem=nullptr;
        if(comp)ANeuralNetworksCompilation_free(comp);comp=nullptr;
        if(nn)ANeuralNetworksModel_free(nn);nn=nullptr;
        if(in)munmap(in,inBytes);if(out)munmap(out,outBytes);in=out=nullptr;
        if(inFd>=0)close(inFd);if(outFd>=0)close(outFd);inFd=outFd=-1;
    }
};

inline double median(std::vector<double> v){std::sort(v.begin(),v.end());return v.empty()?0:v[v.size()/2];}

// Picks the graph for this capture size: the larger model if the NPU runs it within the frame budget,
// otherwise the faster one. Every candidate must match the CPU reference before it is trusted.
inline std::unique_ptr<Net> choose(int W,int H,const std::string& cacheDir,double budgetMs,std::string& log,int onlyModel=-1){
    std::ostringstream s;std::unique_ptr<Net> best;double bestMs=1e9;
    auto devs=accelerators();
    if(devs.empty())throw std::runtime_error("No NNAPI neural accelerator on this device.");
    for(int mi=0;mi<srmodel::MODEL_COUNT;mi++){
        if(onlyModel>=0&&mi!=onlyModel)continue;
        const srmodel::Model& m=*srmodel::MODELS[mi];bool done=false;
        for(int k=0;k<KINDS&&!done;k++)for(const Device& d:devs){
            try{
                auto net=std::make_unique<Net>(m,Kind(k),W,H,d,cacheDir);
                s<<m.name<<" "<<d.name<<" "<<kindName(k)<<": "<<net->log;
                if(!net->supported){s<<"\n";continue;}
                double mean;int worst;double bad=net->verify(&mean,&worst);
                s<<", compile "<<int(net->compileMs)<<" ms, error mean "<<mean<<" max "<<worst;
                if(bad>0.01||mean>1.0){s<<" -> rejected\n";continue;}
                std::vector<double> t;for(int i=0;i<4;i++)t.push_back(net->run());
                double ms=median(t);s<<", "<<ms<<" ms\n";
                if(ms<bestMs){bestMs=ms;best=std::move(net);}
                done=true;break;
            }catch(const std::exception& e){s<<m.name<<" "<<d.name<<" "<<kindName(k)<<": "<<e.what()<<"\n";}
        }
        if(best&&bestMs<=budgetMs)break;
    }
    log=s.str();
    if(!best)throw std::runtime_error("No accelerator ran the super-resolution graph correctly.\n"+log);
    return best;
}

}  // namespace sr
