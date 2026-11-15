#include "engine.h"
#include "flow.h"
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
