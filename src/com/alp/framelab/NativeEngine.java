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
}
