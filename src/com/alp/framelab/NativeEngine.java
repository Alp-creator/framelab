package com.alp.framelab;
final class NativeEngine {
    static { System.loadLibrary("framelab"); }
    static native long create();
    static native String test(long handle);
    static native double generate(long handle,int[] previous,int[] current,int[] output,int width,int height);
    static native void destroy(long handle);
    // GPU pipeline. Buffers are direct: 160x96 RGBA luma in, 20x12 RGBA flow out.
    static native long createFlow();
    static native String flowInfo(long handle);
    static native double flow(long handle,java.nio.ByteBuffer previous,java.nio.ByteBuffer current,java.nio.ByteBuffer flowOut);
    static native void destroyFlow(long handle);
    // x2 AI super resolution on the NPU. Input: width*height luma bytes. Output: width*height*4 detail bytes.
    // model: -1 picks automatically (larger model if it fits budgetMs), otherwise the model index.
    static native long createSr(int width,int height,String cacheDir,double budgetMs,int model);
    static native String srInfo(long handle);
    static native java.nio.ByteBuffer srInput(long handle);
    static native java.nio.ByteBuffer srOutput(long handle);
    static native double srRun(long handle);
    static native void destroySr(long handle);
    static native String srTest(int width,int height,String cacheDir);
}
