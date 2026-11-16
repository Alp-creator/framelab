#include "engine.h"
#include "flow.h"
#include "sr.h"
#include <jni.h>
#include <memory>
static void fail(JNIEnv* env,const std::exception& e){env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),e.what());}
extern "C" JNIEXPORT jlong JNICALL Java_com_alp_framelab_NativeEngine_create(JNIEnv* env,jclass){
    try{auto e=std::make_unique<MotionEngine>();e->init();return reinterpret_cast<jlong>(e.release());}catch(const std::exception& e){fail(env,e);return 0;}
}
extern "C" JNIEXPORT jstring JNICALL Java_com_alp_framelab_NativeEngine_test(JNIEnv* env,jclass,jlong handle){
    try{return env->NewStringUTF(reinterpret_cast<MotionEngine*>(handle)->test().c_str());}catch(const std::exception& e){fail(env,e);return nullptr;}
}
extern "C" JNIEXPORT jdouble JNICALL Java_com_alp_framelab_NativeEngine_generate(JNIEnv* env,jclass,jlong handle,jintArray a,jintArray b,jintArray out,jint w,jint h){
    if(!handle||w<1||h<1||w>1024||h>1024||env->GetArrayLength(a)!=w*h||env->GetArrayLength(b)!=w*h||env->GetArrayLength(out)!=w*h){env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),"Invalid frame dimensions");return 0;}
    std::vector<uint32_t> old(w*h),now(w*h),result(w*h);
    env->GetIntArrayRegion(a,0,w*h,reinterpret_cast<jint*>(old.data()));env->GetIntArrayRegion(b,0,w*h,reinterpret_cast<jint*>(now.data()));
    if(env->ExceptionCheck())return 0;
    try{auto e=reinterpret_cast<MotionEngine*>(handle);e->interpolate(old.data(),now.data(),result.data(),w,h);env->SetIntArrayRegion(out,0,w*h,reinterpret_cast<jint*>(result.data()));return e->lastMs;}catch(const std::exception& e){fail(env,e);return 0;}
}
extern "C" JNIEXPORT void JNICALL Java_com_alp_framelab_NativeEngine_destroy(JNIEnv*,jclass,jlong handle){delete reinterpret_cast<MotionEngine*>(handle);}

// GPU pipeline: motion field only, pixels never leave the GPU.
extern "C" JNIEXPORT jlong JNICALL Java_com_alp_framelab_NativeEngine_createFlow(JNIEnv* env,jclass){
    try{auto e=std::make_unique<FlowEngine>();e->init();return reinterpret_cast<jlong>(e.release());}catch(const std::exception& e){fail(env,e);return 0;}
}
extern "C" JNIEXPORT jstring JNICALL Java_com_alp_framelab_NativeEngine_flowInfo(JNIEnv* env,jclass,jlong handle){
    return env->NewStringUTF(reinterpret_cast<FlowEngine*>(handle)->info().c_str());
}
// Returns NPU milliseconds, or -1 when the frames are unrelated (scene cut).
extern "C" JNIEXPORT jdouble JNICALL Java_com_alp_framelab_NativeEngine_flow(JNIEnv* env,jclass,jlong handle,jobject a,jobject b,jobject out){
    auto pa=static_cast<uint8_t*>(env->GetDirectBufferAddress(a)),pb=static_cast<uint8_t*>(env->GetDirectBufferAddress(b)),po=static_cast<uint8_t*>(env->GetDirectBufferAddress(out));
    const jlong need=FlowEngine::LW*FlowEngine::LH*4,flowBytes=FlowEngine::FX*FlowEngine::FY*4;
    if(!handle||!pa||!pb||!po||env->GetDirectBufferCapacity(a)<need||env->GetDirectBufferCapacity(b)<need||env->GetDirectBufferCapacity(out)<flowBytes){env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),"Invalid flow buffers");return 0;}
    try{auto e=reinterpret_cast<FlowEngine*>(handle);return e->flow(pa,pb,po)?e->npuMs:-1.0;}catch(const std::exception& e){fail(env,e);return 0;}
}
extern "C" JNIEXPORT void JNICALL Java_com_alp_framelab_NativeEngine_destroyFlow(JNIEnv*,jclass,jlong handle){delete reinterpret_cast<FlowEngine*>(handle);}

// x2 AI super resolution: the GPU reads luma straight into the NPU's input memory and uploads the
// detail straight from its output memory; both are direct ByteBuffers over the same shared mapping.
struct SrHandle{std::unique_ptr<sr::Net> net;std::string log;};
static std::string jstr(JNIEnv* env,jstring s){if(!s)return "";const char* c=env->GetStringUTFChars(s,nullptr);std::string r=c?c:"";if(c)env->ReleaseStringUTFChars(s,c);return r;}
extern "C" JNIEXPORT jlong JNICALL Java_com_alp_framelab_NativeEngine_createSr(JNIEnv* env,jclass,jint w,jint h,jstring cache,jdouble budget,jint model){
    if(w<8||h<8||w>4096||h>4096||w%4){env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),"Invalid super-resolution size");return 0;}
    try{auto s=std::make_unique<SrHandle>();s->net=sr::choose(w,h,jstr(env,cache),budget,s->log,model);return reinterpret_cast<jlong>(s.release());}catch(const std::exception& e){fail(env,e);return 0;}
}
extern "C" JNIEXPORT jstring JNICALL Java_com_alp_framelab_NativeEngine_srInfo(JNIEnv* env,jclass,jlong handle){
    auto s=reinterpret_cast<SrHandle*>(handle);return env->NewStringUTF((s->net->describe()+"\n"+s->log).c_str());
}
extern "C" JNIEXPORT jobject JNICALL Java_com_alp_framelab_NativeEngine_srInput(JNIEnv* env,jclass,jlong handle){
    auto n=reinterpret_cast<SrHandle*>(handle)->net.get();return env->NewDirectByteBuffer(n->in,jlong(n->inBytes));
}
extern "C" JNIEXPORT jobject JNICALL Java_com_alp_framelab_NativeEngine_srOutput(JNIEnv* env,jclass,jlong handle){
    auto n=reinterpret_cast<SrHandle*>(handle)->net.get();return env->NewDirectByteBuffer(n->out,jlong(n->outBytes));
}
extern "C" JNIEXPORT jdouble JNICALL Java_com_alp_framelab_NativeEngine_srRun(JNIEnv* env,jclass,jlong handle){
    try{return reinterpret_cast<SrHandle*>(handle)->net->run();}catch(const std::exception& e){fail(env,e);return 0;}
}
extern "C" JNIEXPORT void JNICALL Java_com_alp_framelab_NativeEngine_destroySr(JNIEnv*,jclass,jlong handle){delete reinterpret_cast<SrHandle*>(handle);}
// In-app self-test: the graph chosen for this screen, and the same graph at 4K output sizes.
extern "C" JNIEXPORT jstring JNICALL Java_com_alp_framelab_NativeEngine_srTest(JNIEnv* env,jclass,jint w,jint h,jstring cache){
    try{
        std::string log;auto net=sr::choose(w,h,jstr(env,cache),20.0,log);
        std::ostringstream s;s<<"device="<<net->dev.name<<"\ngraph="<<sr::kindName(net->kind)<<", model "<<net->model.name<<" ("<<net->model.macsPerPixel<<" MAC/px)\nCPU fallback=disabled\n";
        auto bench=[&](sr::Net& n){sr::testPattern(n.in,n.W,n.H);std::vector<double> t;for(int i=0;i<10;i++)t.push_back(n.run());t.erase(t.begin(),t.begin()+2);return sr::median(t);};
        double ms=bench(*net);
        s<<w<<"x"<<h<<" -> "<<2*w<<"x"<<2*h<<": "<<ms<<" ms per frame ("<<int(1000/ms)<<" fps)\n";
        const int sizes[][2]={{1920,1080},{1920,1200}};
        for(auto& z:sizes){
            if(z[0]==w&&z[1]==h)continue;
            try{sr::Net n(net->model,net->kind,z[0],z[1],net->dev);double t=bench(n);s<<z[0]<<"x"<<z[1]<<" -> "<<2*z[0]<<"x"<<2*z[1]<<" (4K): "<<t<<" ms ("<<int(1000/t)<<" fps)\n";}
            catch(const std::exception& e){s<<z[0]<<"x"<<z[1]<<": "<<e.what()<<"\n";}
        }
        s<<"\ncandidates:\n"<<log;
        return env->NewStringUTF(s.str().c_str());
    }catch(const std::exception& e){fail(env,e);return nullptr;}
}
