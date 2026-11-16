// On-device check of the x2 super-resolution graphs: which accelerator runs which variant, whether the
// result matches the CPU reference, how long it takes (including the 4K output sizes), and, given a
// high-res luma frame, how much closer to it the NPU result is than bilinear upscaling.
//
//   sr-test                         device/variant table and 4K benchmark
//   sr-test frame.y 2944 1840       + quality on that frame (raw 8-bit luma), writes /data/local/tmp/sr-out.y
#include "../native/sr.h"
#include <cstdio>
#include <fstream>
#include <map>

static double psnr(const std::vector<double>& a,const std::vector<uint8_t>& b){
    double mse=0;for(size_t i=0;i<a.size();i++){double d=std::clamp(a[i],0.0,255.0)-b[i];mse+=d*d;}
    mse/=a.size();return 10*std::log10(255.0*255.0/std::max(mse,1e-9));
}
// Bilinear x2 with half-pixel centres and edge clamp: what the GPU's texture filtering draws.
static double bilinear(const uint8_t* lr,int W,int H,int X,int Y){
    double sx=X*0.5-0.25,sy=Y*0.5-0.25;int x0=int(std::floor(sx)),y0=int(std::floor(sy));double fx=sx-x0,fy=sy-y0;
    auto at=[&](int x,int y){return double(lr[std::clamp(y,0,H-1)*W+std::clamp(x,0,W-1)]);};
    return (at(x0,y0)*(1-fx)+at(x0+1,y0)*fx)*(1-fy)+(at(x0,y0+1)*(1-fx)+at(x0+1,y0+1)*fx)*fy;
}

int main(int argc,char** argv){
    try{
        auto devs=sr::accelerators();
        std::printf("accelerators:");for(auto& d:devs)std::printf(" %s",d.name.c_str());std::printf("\n");
        const int W=1472,H=920; // half of the TB373FU panel: output 2944x1840
        struct Best{int kind=-1;size_t dev=0;double ms=1e9;};std::map<int,Best> best;
        for(int mi=0;mi<srmodel::MODEL_COUNT;mi++){
            const auto& m=*srmodel::MODELS[mi];
            std::printf("\n%s (%d MAC/px, %s)\n",m.name,m.macsPerPixel,m.info);
            for(int k=0;k<sr::KINDS;k++)for(size_t di=0;di<devs.size();di++){
                std::printf("  %-16s %-17s ",devs[di].name.c_str(),sr::kindName(k));std::fflush(stdout);
                try{
                    sr::Net net(m,sr::Kind(k),W,H,devs[di]);
                    if(!net.supported){std::printf("%s: not supported\n",net.log.c_str());continue;}
                    double mean;int worst;double bad=net.verify(&mean,&worst);
                    std::vector<double> t,hw,drv;for(int i=0;i<12;i++){t.push_back(net.run());hw.push_back(net.hwMs);drv.push_back(net.driverMs);}
                    t.erase(t.begin(),t.begin()+2);
                    bool ok=bad<=0.01&&mean<=1.0;
                    std::printf("compile %4.0f ms | %6.2f ms (hw %.2f, driver %.2f) | err mean %.3f max %d >3: %.2f%% %s | %s\n",
                        net.compileMs,sr::median(t),sr::median(hw),sr::median(drv),mean,worst,100*bad,ok?"OK":"WRONG",net.log.c_str());
                    if(ok&&sr::median(t)<best[mi].ms)best[mi]={k,di,sr::median(t)};
                }catch(const std::exception& e){std::printf("%s\n",e.what());}
            }
        }
        std::printf("\nbenchmark, best variant per model (input -> output):\n");
        const int sizes[][2]={{1472,920},{1920,1080},{1920,1200}};
        for(auto& [mi,b]:best){
            const auto& m=*srmodel::MODELS[mi];
            for(auto& s:sizes){
                try{
                    sr::Net net(m,sr::Kind(b.kind),s[0],s[1],devs[b.dev]);
                    sr::testPattern(net.in,s[0],s[1]);
                    std::vector<double> t;for(int i=0;i<15;i++)t.push_back(net.run());t.erase(t.begin(),t.begin()+3);
                    double ms=sr::median(t);
                    std::printf("  %s %-16s %-17s %4dx%-4d -> %4dx%-4d %6.2f ms  (%.0f fps)%s\n",m.name,devs[b.dev].name.c_str(),sr::kindName(b.kind),
                        s[0],s[1],2*s[0],2*s[1],ms,1000/ms,s[0]*2>=3840?"  4K":"");
                }catch(const std::exception& e){std::printf("  %s %dx%d: %s\n",m.name,s[0],s[1],e.what());}
            }
        }
        if(argc>=4&&!best.empty()){
            const int HW=std::atoi(argv[2]),HH=std::atoi(argv[3]),w=HW/2/4*4,h=HH/2/2*2;
            std::vector<uint8_t> hr(size_t(HW)*HH);std::ifstream f(argv[1],std::ios::binary);f.read(reinterpret_cast<char*>(hr.data()),hr.size());
            if(!f)throw std::runtime_error(std::string("cannot read ")+argv[1]);
            std::printf("\nquality on %s (%dx%d; low-res %dx%d = 2x2 average):\n",argv[1],HW,HH,w,h);
            std::vector<uint8_t> truth(size_t(4)*w*h);
            for(int y=0;y<2*h;y++)for(int x=0;x<2*w;x++)truth[size_t(y)*2*w+x]=hr[size_t(y)*HW+x];
            for(auto& [mi,b]:best){
                const auto& m=*srmodel::MODELS[mi];
                sr::Net net(m,sr::Kind(b.kind),w,h,devs[b.dev]);
                for(int y=0;y<h;y++)for(int x=0;x<w;x++){int s=0;for(int k=0;k<4;k++)s+=hr[size_t(2*y+k/2)*HW+2*x+k%2];net.in[size_t(y)*w+x]=uint8_t((s+2)/4);}
                double ms=net.run();
                std::vector<double> lin(truth.size()),ai(truth.size());
                for(int y=0;y<2*h;y++)for(int x=0;x<2*w;x++){
                    size_t i=size_t(y)*2*w+x;lin[i]=bilinear(net.in,w,h,x,y);
                    ai[i]=lin[i]+double(int(net.out[(size_t(y/2)*w+x/2)*4+(y%2)*2+x%2])-128);
                }
                std::printf("  %s on %s (%s, %.1f ms): bilinear %.2f dB, AI super resolution %.2f dB (%+.2f dB)\n",m.name,devs[b.dev].name.c_str(),
                    sr::kindName(b.kind),ms,psnr(lin,truth),psnr(ai,truth),psnr(ai,truth)-psnr(lin,truth));
                std::vector<uint8_t> img(ai.size());for(size_t i=0;i<ai.size();i++)img[i]=uint8_t(std::clamp(std::lround(ai[i]),0l,255l));
                std::string name=std::string("/data/local/tmp/sr-out-")+m.name+".y";
                std::ofstream(name,std::ios::binary).write(reinterpret_cast<char*>(img.data()),img.size());
                std::printf("  wrote %s (%dx%d)\n",name.c_str(),2*w,2*h);
            }
        }
        return 0;
    }catch(const std::exception& e){std::printf("FAILED: %s\n",e.what());return 1;}
}
