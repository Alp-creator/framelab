package com.alp.framelab;
import android.app.*;
import android.content.*;
import android.content.pm.ServiceInfo;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.display.*;
import android.media.projection.*;
import android.opengl.*;
import android.os.*;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Game frames stay on the GPU the whole way: screen capture -> texture -> warp shader -> overlay.
 * Frame generation reads back only a 160x96 luma thumbnail, for the NPU motion search.
 * Super resolution reads back the half-size luma (1 byte per pixel) for the NPU network and uploads
 * its detail (4 bytes per low-res pixel); colour is upscaled on the GPU.
 */
public class CaptureService extends Service implements SurfaceHolder.Callback {
    static volatile boolean active=false;
    static final int LW=160,LH=96,FW=20,FH=12,LONG_SIDE=4096; // no downscale: capture at the full screen resolution
    static final long MAX_GAP_NS=150_000_000L,MIN_GAP_NS=12_000_000L;
    final Handler main=new Handler(Looper.getMainLooper());
    HandlerThread thread;Handler worker;MediaProjection projection;VirtualDisplay display;SurfaceTexture input;Surface inputSurface;
    volatile boolean stopping=false;long engine=0;String device="?";
    // EGL / GL state, worker thread only.
    EGLDisplay egl=EGL14.EGL_NO_DISPLAY;EGLContext context=EGL14.EGL_NO_CONTEXT;EGLSurface window=EGL14.EGL_NO_SURFACE;EGLConfig config;
    int screenW,screenH,capW,capH,oesTex,smallTex,smallFbo,flowTex,pOes,pDown,pShow,pMid;
    final int[] tex=new int[3],fbo=new int[3];int prev=0,cur=1,incoming=2;
    final ByteBuffer[] luma=new ByteBuffer[3];final ByteBuffer flow=ByteBuffer.allocateDirect(FW*FH*4);
    final float[] stMatrix=new float[16],identity=new float[16];FloatBuffer quad;
    boolean have=false;long lastTime=0,firstBlack=0,statsEpoch=0;int realCount=0,midCount=0,cutCount=0;double npuSum=0,workSum=0,readSum=0,texSum=0;int readCount=0;boolean highPriority=false;
    boolean realPending=false;
    // Overload guard: when generating a frame takes most of the frame interval it only adds lag,
    // so generation pauses (frames pass straight through) and is retried later.
    static final boolean GUARD=false; // overload guard switched off on request: generation never pauses, even if the game slows down
    static final long RETRY_MIN_NS=20_000_000_000L,RETRY_MAX_NS=60_000_000_000L;
    long bypassUntil=0,retryNs=RETRY_MIN_NS;int slow=0,healthy=0,sinceRetry=0;final StringBuilder trace=new StringBuilder();
    double baseGap=0; // the game's own frame interval, measured while frames pass straight through
    int longSide=LONG_SIDE;
    // Modes: NPU frame generation, x2 AI super resolution on the NPU, capture passthrough.
    static final int FG=0,SR=1,OFF=2;static final String[] MODE_NAMES={"NPU","SR","OFF"};
    static volatile int requestedBackend=FG;
    static void setMode(Context c,int b){requestedBackend=b;try{java.nio.file.Files.write(new File(c.getFilesDir(),"backend").toPath(),MODE_NAMES[b].getBytes(java.nio.charset.StandardCharsets.UTF_8));}catch(Exception e){android.util.Log.w("FrameLab","mode",e);}}
    // Super resolution: the game is captured at half size per axis, the NPU predicts the missing detail and
    // the GPU draws bilinear colour + that detail at full size. The larger network is used while it needs
    // at most SR_BUDGET_MS of NPU time per frame, otherwise the faster one.
    static final double SR_BUDGET_MS=20;
    static volatile boolean srCompare=false; // left half plain bilinear, right half AI (long-press MODE)
    long sr=0;int srW,srH,srCount,lumaTex,lumaFbo,resTex,pLuma,pSr,fgW,fgH,contentW,contentH;ByteBuffer srIn,srOut;byte[] srLast;boolean srHave=false;double uploadSum=0;String srInfo="",srError="",srWarning="";
    // A failed NPU pass rebuilds the graph; only repeated failures within SR_FAIL_WINDOW_NS end SR (the capture keeps running).
    static final int SR_MAX_FAILURES=3;static final long SR_FAIL_WINDOW_NS=30_000_000_000L;int srFailures=0;long srFailSince=0;
    int backend=-1;long backendPoll=0;String mode="NPU";
    java.io.FileOutputStream metrics;
    int inputCount=0;double flowSum=0;
    // A capture counts as a new game frame only if enough of the thumbnail really changed.
    // Emulators and HUD overlays re-present the same game frame many times with tiny differences;
    // treating those as frames wastes GPU time and leaves nothing to interpolate.
    static final int DIFF=4,NEW_FRAME_PIXELS=160; /* about 1% of the thumbnail: more than a small HUD or counter overlay changes */ static final long STALE_NS=120_000_000L;
    int repeatCount=0,newMin=Integer.MAX_VALUE;long repeatSum=0,newSum=0;
    static final int[] BUCKETS={1,8,24,48,100,200,400,800,1600,3200,6400,Integer.MAX_VALUE};final int[] histogram=new int[BUCKETS.length];
    static int changed(ByteBuffer a,ByteBuffer b){int n=0;for(int k=0;k<LW*LH*4;k+=4){int d=(a.get(k)&255)-(b.get(k)&255);if(d>DIFF||d<-DIFF)n++;}return n;}
    long[] engines=new long[2];
    final Runnable showReal=()->{realPending=false;if(!stopping&&window!=EGL14.EGL_NO_SURFACE){present(pShow);realCount++;}};

    @Override public IBinder onBind(Intent intent){return null;}
    @Override public int onStartCommand(Intent i,int flags,int id){
        if(i==null||"stop".equals(i.getAction())){stopSelf();return START_NOT_STICKY;}if(active)return START_NOT_STICKY;active=true;
        try{
            NotificationManager nm=getSystemService(NotificationManager.class);nm.createNotificationChannel(new NotificationChannel("capture","FrameLab capture",NotificationManager.IMPORTANCE_LOW));
            PendingIntent stop=PendingIntent.getService(this,1,new Intent(this,CaptureService.class).setAction("stop"),PendingIntent.FLAG_IMMUTABLE);
            PendingIntent open=PendingIntent.getActivity(this,2,new Intent(this,MainActivity.class),PendingIntent.FLAG_IMMUTABLE);
            Notification n=new Notification.Builder(this,"capture").setSmallIcon(android.R.drawable.ic_media_play).setContentTitle("FrameLab is running").setContentText("The screen is processed on this device only · Tap to stop").setContentIntent(open).addAction(new Notification.Action.Builder(null,"Stop",stop).build()).setOngoing(true).build();
            startForeground(7,n,ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
            OverlayService overlay=OverlayService.instance;if(overlay==null)throw new IllegalStateException("The overlay service is off.");
            // Normal priority: emulated games are often CPU-bound, and a higher priority here takes CPU time from the game itself.
            thread=new HandlerThread("FrameLab-gl",android.os.Process.THREAD_PRIORITY_DEFAULT);thread.start();worker=new Handler(thread.getLooper());
            Intent token=i.getParcelableExtra("token",Intent.class);
            projection=getSystemService(MediaProjectionManager.class).getMediaProjection(i.getIntExtra("result",0),token);
            projection.registerCallback(new MediaProjection.Callback(){
                @Override public void onStop(){android.util.Log.i("FrameLab","screen capture stopped by the system");main.post(()->stopSelf());}
                @Override public void onCapturedContentResize(int w,int h){worker.post(()->{try{resize(w,h);}catch(Throwable e){error(e);}});}
            },main);
            // Developer switch for screenshots of the output: run-as com.alp.framelab touch files/insecure
            SurfaceView view;
            try{view=overlay.show(!new File(getFilesDir(),"insecure").exists());}
            catch(android.view.WindowManager.BadTokenException e){overlay.hide();throw new IllegalStateException("The overlay service is not ready yet. Wait a few seconds and try again.");}
            view.getHolder().addCallback(this);
        }catch(Throwable e){error(e);}
        return START_NOT_STICKY;
    }

    // ---- output surface lifecycle (main thread) ----
    @Override public void surfaceCreated(SurfaceHolder h){}
    @Override public void surfaceChanged(SurfaceHolder h,int format,int w,int height){
        Surface s=h.getSurface();worker.post(()->{try{attach(s,w,height);}catch(Throwable e){error(e);}});
    }
    @Override public void surfaceDestroyed(SurfaceHolder h){
        CountDownLatch done=new CountDownLatch(1);
        if(worker==null||!worker.post(()->{try{releaseWindow();}finally{done.countDown();}}))return;
        try{done.await(700,TimeUnit.MILLISECONDS);}catch(InterruptedException ignored){}
    }

    // ---- everything below runs on the GL worker thread ----
    void attach(Surface surface,int w,int h){
        if(stopping)return;
        if(egl==EGL14.EGL_NO_DISPLAY){
            egl=EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);int[] v=new int[2];
            if(!EGL14.eglInitialize(egl,v,0,v,1))throw new IllegalStateException("EGL could not be initialised");
            int[] attr={EGL14.EGL_RED_SIZE,8,EGL14.EGL_GREEN_SIZE,8,EGL14.EGL_BLUE_SIZE,8,EGL14.EGL_RENDERABLE_TYPE,0x40,EGL14.EGL_SURFACE_TYPE,EGL14.EGL_WINDOW_BIT,EGL14.EGL_NONE};
            EGLConfig[] configs=new EGLConfig[1];int[] count=new int[1];
            if(!EGL14.eglChooseConfig(egl,attr,0,configs,0,1,count,0)||count[0]<1)throw new IllegalStateException("No matching EGL configuration");
            config=configs[0];
            // Ask the GPU driver to schedule our small jobs ahead of the game's (EGL_IMG_context_priority, HIGH).
            context=EGL14.eglCreateContext(egl,config,EGL14.EGL_NO_CONTEXT,new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION,3,0x3100,0x3101,EGL14.EGL_NONE},0);
            highPriority=context!=EGL14.EGL_NO_CONTEXT;
            if(!highPriority)context=EGL14.eglCreateContext(egl,config,EGL14.EGL_NO_CONTEXT,new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION,3,EGL14.EGL_NONE},0);
            if(context==EGL14.EGL_NO_CONTEXT)throw new IllegalStateException("The GL context could not be created");
        }
        releaseWindow();
        window=EGL14.eglCreateWindowSurface(egl,config,surface,new int[]{EGL14.EGL_NONE},0);
        if(window==EGL14.EGL_NO_SURFACE||!EGL14.eglMakeCurrent(egl,window,window,context))throw new IllegalStateException("The output surface could not be attached");
        EGL14.eglSwapInterval(egl,0); // never block the worker on vsync
        screenW=w;screenH=h;
        if(input==null)start();else if(have)present(pShow);
    }
    void releaseWindow(){
        if(window!=EGL14.EGL_NO_SURFACE){EGL14.eglMakeCurrent(egl,EGL14.EGL_NO_SURFACE,EGL14.EGL_NO_SURFACE,context);EGL14.eglDestroySurface(egl,window);window=EGL14.EGL_NO_SURFACE;}
    }
    void start(){
        chooseBackend(requestedBackend);
        try{metrics=new java.io.FileOutputStream(new File(getFilesDir(),"comparison.csv"));metrics.write("elapsed_ns,mode,bypass,captured_fps,real_submitted_fps,mid_submitted_fps,accelerator_ms,flow_ms,work_ms,read_ms,texture_ms,width,height,upload_ms\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));}catch(Exception e){throw new IllegalStateException(e);}
        bypassUntil=GUARD?System.nanoTime()+2_000_000_000L:0; // first two seconds: only measure the game's pace
        // Developer override of the capture size: run-as com.alp.framelab sh -c "echo 1472 > files/longside"
        try{longSide=Math.max(320,Integer.parseInt(new String(java.nio.file.Files.readAllBytes(new File(getFilesDir(),"longside").toPath())).trim()));}catch(Exception ignored){}
        android.opengl.Matrix.setIdentityM(identity,0);
        quad=ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer();quad.put(new float[]{-1,-1,1,-1,-1,1,1,1}).position(0);
        for(int k=0;k<3;k++)luma[k]=ByteBuffer.allocateDirect(LW*LH*4);
        String head="precision highp float;varying vec2 uv;";
        pOes=program("#extension GL_OES_EGL_image_external : require\n"+head+"uniform samplerExternalOES t;void main(){gl_FragColor=texture2D(t,uv);}");
        // 4x4 box filter: an honest thumbnail instead of aliased point samples.
        pDown=program("#extension GL_OES_EGL_image_external : require\n"+head+"uniform samplerExternalOES t;const vec2 f=vec2(1.0/160.0,1.0/96.0);void main(){vec3 s=vec3(0.0);for(int j=0;j<4;j++)for(int i=0;i<4;i++)s+=texture2D(t,uv+(vec2(float(i),float(j))-1.5)*0.25*f).rgb;float l=dot(s/16.0,vec3(0.299,0.587,0.114));gl_FragColor=vec4(l,l,l,1.0);}");
        pShow=program(head+"uniform sampler2D t;void main(){gl_FragColor=texture2D(t,uv);}");
        // Midpoint with per-pixel candidate selection. The block motion field is coarse, so around a
        // moving object it also drags static background along (halos). Each pixel therefore tries the
        // interpolated motion, its own block's motion and no motion, and trusts whichever makes the
        // two neighbouring frames agree. Where none agrees (newly revealed area) it shows the newer
        // frame instead of a ghosted blend.
        pMid=program(head+"uniform sampler2D t,b,fl;"
            +"vec2 mv(vec2 q){return (texture2D(fl,q).rg*255.0-128.0)/4.0/vec2(160.0,96.0)*0.5;}"
            +"void main(){vec2 g=vec2(20.0,12.0);vec2 v1=mv(uv);vec2 v2=mv((floor(uv*g)+0.5)/g);"
            +"vec3 a0=texture2D(t,uv).rgb,b0=texture2D(b,uv).rgb,a1=texture2D(t,uv+v1).rgb,b1=texture2D(b,uv-v1).rgb,a2=texture2D(t,uv+v2).rgb,b2=texture2D(b,uv-v2).rgb;"
            +"float d0=length(a0-b0),d1=length(a1-b1),d2=length(a2-b2);"
            +"float w0=1.0/(d0*d0+0.0004),w1=1.0/(d1*d1+0.0004),w2=1.0/(d2*d2+0.0004);"
            +"vec3 c=(w0*(a0+b0)+w1*(a1+b1)+w2*(a2+b2))*0.5/(w0+w1+w2);"
            +"gl_FragColor=vec4(mix(c,b0,smoothstep(0.12,0.35,min(d0,min(d1,d2)))),1.0);}");
        // Super resolution, step 1: low-res luma, four horizontal neighbours per RGBA texel (the NPU's input bytes).
        pLuma=program("#extension GL_OES_EGL_image_external : require\n"+head+"uniform samplerExternalOES t;uniform vec2 du;"
            +"float y(vec2 q){return dot(texture2D(t,q).rgb,vec3(0.299,0.587,0.114));}"
            +"void main(){gl_FragColor=vec4(y(uv-1.5*du),y(uv-0.5*du),y(uv+0.5*du),y(uv+1.5*du));}");
        // Super resolution, step 2: bilinear colour plus the NPU's detail for this pixel's 2x2 phase.
        // Adding the same value to R, G and B changes luma only; colour stays the GPU's bilinear upscale.
        pSr=program("attribute vec2 p;varying vec2 uv;varying vec2 pos;uniform mat4 m;void main(){pos=p*0.5+0.5;uv=(m*vec4(pos,0.0,1.0)).xy;gl_Position=vec4(p,0.0,1.0);}",
            "#extension GL_OES_EGL_image_external : require\nprecision highp float;varying vec2 uv;varying vec2 pos;"
            +"uniform samplerExternalOES t;uniform sampler2D r;uniform vec2 lr;uniform float split;"
            +"void main(){vec2 h=pos*lr*2.0;vec2 q=floor(h*0.5);vec2 f=floor(h)-2.0*q;"
            +"float d=dot(texture2D(r,(q+0.5)/lr),vec4((1.0-f.x)*(1.0-f.y),f.x*(1.0-f.y),(1.0-f.x)*f.y,f.x*f.y))-128.0/255.0;"
            +"if(pos.x<split)d=0.0;"
            +"vec3 c=texture2D(t,uv).rgb+d;"
            +"gl_FragColor=vec4(abs(pos.x-split)*lr.x*2.0<1.5?vec3(1.0):c,1.0);}");
        int[] id=new int[1];
        GLES20.glGenTextures(1,id,0);oesTex=id[0];GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,oesTex);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE);GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glGenTextures(3,tex,0);GLES20.glGenFramebuffers(3,fbo,0);
        smallTex=texture(LW,LH,null);GLES20.glGenFramebuffers(1,id,0);smallFbo=id[0];bind(smallFbo,smallTex);
        flowTex=texture(FW,FH,null);
        Rect bounds=getSystemService(WindowManager.class).getMaximumWindowMetrics().getBounds();
        contentW=bounds.width();contentH=bounds.height();size(contentW,contentH);if(backend!=SR)allocate();
        input=new SurfaceTexture(oesTex);input.setDefaultBufferSize(capW,capH);input.setOnFrameAvailableListener(st->frame(),worker);
        inputSurface=new Surface(input);
        display=projection.createVirtualDisplay("FrameLab single-app",capW,capH,getResources().getDisplayMetrics().densityDpi,DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,inputSurface,null,worker);
        status(String.format(Locale.US,"FrameLab • %s ready • %dx%d",device,capW,capH),false);
        android.util.Log.i("FrameLab","started device="+device+" capture="+capW+"x"+capH+" screen="+screenW+"x"+screenH);
    }
    void size(int w,int h){
        if(backend==SR){capW=Math.max(8,w/2/4*4);capH=Math.max(8,h/2/2*2);return;} // half size per axis; width a multiple of 4 for the luma packing
        float s=Math.min(1f,(float)longSide/Math.max(w,h));capW=Math.max(2,Math.round(w*s)/2*2);capH=Math.max(2,Math.round(h*s)/2*2);
    }
    void chooseBackend(int requested){
        if(requested<FG||requested>OFF)throw new IllegalArgumentException("Unknown backend");
        worker.removeCallbacks(showReal);realPending=false;
        if(requested==FG){if(engines[0]==0)engines[0]=NativeEngine.createFlow();engine=engines[0];device=NativeEngine.flowInfo(engine);}
        else{engine=0;device=requested==SR?"AI super resolution":"capture passthrough";}
        if(requested!=OFF)srError="";
        backend=requested;mode=MODE_NAMES[backend];have=false;srHave=false;lastTime=0;baseGap=0;
        bypassUntil=GUARD?System.nanoTime()+2_000_000_000L:0;retryNs=RETRY_MIN_NS;slow=healthy=sinceRetry=0;
        statsEpoch=0;realCount=midCount=cutCount=readCount=inputCount=srCount=0;npuSum=flowSum=workSum=readSum=texSum=uploadSum=0;
        android.util.Log.i("FrameLab","BACKEND "+mode+" "+device);
        if(display!=null)applyCapture();
    }
    void pollBackend(long now){
        if(now-backendPoll<500_000_000L)return;backendPoll=now;int request=requestedBackend;
        try{String file=new String(java.nio.file.Files.readAllBytes(new File(getFilesDir(),"backend").toPath())).trim();request=file.equals("OFF")?OFF:file.equals("SR")?SR:FG;}catch(Exception ignored){}
        if(request!=backend)chooseBackend(request);
    }
    void allocate(){
        for(int k=0;k<3;k++){
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,tex[k]);GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D,0,GLES20.GL_RGBA,capW,capH,0,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,null);
            params();bind(fbo[k],tex[k]);
        }
        fgW=capW;fgH=capH;have=false;lastTime=0;
    }
    void resize(int w,int h){
        if(stopping||display==null||w<=0||h<=0)return;
        contentW=w;contentH=h;applyCapture();
    }
    /** The capture size follows the mode: full resolution for frame generation, half for super resolution. */
    void applyCapture(){
        int oldW=capW,oldH=capH;size(contentW,contentH);
        if(backend!=SR&&(fgW!=capW||fgH!=capH))allocate();
        if(capW==oldW&&capH==oldH)return;
        worker.removeCallbacks(showReal);realPending=false;have=false;srHave=false;
        input.setDefaultBufferSize(capW,capH);display.resize(capW,capH,getResources().getDisplayMetrics().densityDpi);
        android.util.Log.i("FrameLab","capture resized to "+capW+"x"+capH);
    }
    void frame(){
        if(stopping||input==null)return;
        try{
            long now=System.nanoTime();
            pollBackend(now);
            // A real frame still waiting for its slot is shown now, never skipped.
            if(realPending){worker.removeCallbacks(showReal);showReal.run();}
            input.updateTexImage();input.getTransformMatrix(stMatrix);long acquired=System.nanoTime();
            if(backend==OFF){inputCount++;passThrough();realCount++;stats(now);return;}
            if(backend==SR){superRes(now,acquired);return;}
            if(bypassUntil!=0){
                if(now<bypassUntil){
                    long g=now-lastTime;if(lastTime!=0&&g<MAX_GAP_NS)baseGap=baseGap==0?g:baseGap*.9+g*.1; // true game pace, unloaded
                    lastTime=now;passThrough();realCount++;stats(now);return;
                }
                bypassUntil=0;slow=0;healthy=0;sinceRetry=0;trace.setLength(0);have=false;lastTime=0;android.util.Log.i("FrameLab","retrying frame generation");
            }
            // 1. Thumbnail straight from the captured frame: the only GPU result the CPU waits for.
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,oesTex);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,smallFbo);GLES20.glViewport(0,0,LW,LH);GLES20.glUseProgram(pDown);draw(pDown,stMatrix);
            ByteBuffer small=luma[incoming];small.position(0);GLES20.glReadPixels(0,0,LW,LH,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,small);
            readSum+=(System.nanoTime()-acquired)/1e6;texSum+=(acquired-now)/1e6;readCount++;
            int moved=have?changed(small,luma[cur]):Integer.MAX_VALUE;boolean refresh=false;
            if(have)for(int h=0;h<BUCKETS.length;h++)if(moved<BUCKETS[h]){histogram[h]++;break;}
            if(moved<NEW_FRAME_PIXELS){
                repeatCount++;repeatSum+=moved;
                if(now-lastTime<STALE_NS)return; // same game frame again: nothing to draw
                refresh=true;                    // slow change: keep the picture current, without a generated frame
            }else if(have){newSum+=moved;newMin=Math.min(newMin,moved);}
            inputCount++;
            // 2. Full-resolution copy into our own texture; the GPU does it while the NPU works.
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,fbo[incoming]);GLES20.glViewport(0,0,capW,capH);
            GLES20.glUseProgram(pOes);draw(pOes,stMatrix);GLES20.glFlush();
            int bright=0;for(int k=0;k<LW*LH*4;k+=4*37)bright+=small.get(k)&255;
            if(bright<60){if(firstBlack==0)firstBlack=now;if(now-firstBlack>4_000_000_000L)throw new IllegalStateException("The capture is black. Choose single-app sharing, or try a game without protected content.");}else firstBlack=0;
            int old=prev;prev=cur;cur=incoming;incoming=old;
            long gap=now-lastTime;boolean mid=false;
            if(have&&!refresh&&gap<MAX_GAP_NS&&gap>MIN_GAP_NS&&window!=EGL14.EGL_NO_SURFACE){
                // 3. NPU: where did every block of the new frame come from?
                long flowStart=System.nanoTime();double npu=NativeEngine.flow(engine,luma[prev],luma[cur],flow);
                if(npu>=0){
                    flowSum+=(System.nanoTime()-flowStart)/1e6;
                    GLES20.glActiveTexture(GLES20.GL_TEXTURE2);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,flowTex);flow.position(0);
                    GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D,0,GLES20.GL_RGBA,FW,FH,0,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,flow);
                    // 4. Show the generated frame now, the real one half an interval later.
                    present(pMid);mid=true;midCount++;npuSum+=npu;long spent=System.nanoTime()-now;workSum+=spent/1e6;
                    // Half an interval later. If the next game frame arrives first, it flushes this one.
                    long wait=gap/2;realPending=true;
                    if(baseGap==0)baseGap=gap;
                    // The first frames after a pause are slow while NPU and GPU clocks ramp up: do not judge them.
                    if(++sinceRetry<=60)trace.append(spent/1_000_000L).append('/').append(gap/1_000_000L).append(' ');
                    if(!GUARD||sinceRetry<=20){}
                    else if(spent>baseGap*.9||gap>baseGap*1.15){healthy=0;if(++slow>=8){android.util.Log.i("FrameLab","trip after "+sinceRetry+" frames, base "+(long)(baseGap/1e6)+" ms, work/gap ms: "+trace);bypassUntil=now+retryNs;retryNs=Math.min(retryNs*2,RETRY_MAX_NS);android.util.Log.i("FrameLab","overloaded, pausing generation for "+(bypassUntil-now)/1_000_000_000L+" s");}}
                    else{if(slow>0)slow--;if(++healthy>=900)retryNs=RETRY_MIN_NS;}
                    worker.postDelayed(showReal,Math.max(1,wait/1_000_000L));
                }else cutCount++;
            }
            if(!mid)showReal.run();
            have=true;lastTime=now;stats(now);
        }catch(Throwable e){error(e);}
    }
    void screen(){
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,0);GLES20.glViewport(0,0,screenW,screenH);GLES20.glClearColor(0,0,0,1);GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        float scale=Math.min((float)screenW/capW,(float)screenH/capH);int w=Math.round(capW*scale),h=Math.round(capH*scale);
        GLES20.glViewport((screenW-w)/2,(screenH-h)/2,w,h);
    }
    /** Overload mode: the captured frame goes straight to the screen, one draw, no readback, no NPU. */
    void passThrough(){
        if(window==EGL14.EGL_NO_SURFACE)return;
        screen();GLES20.glUseProgram(pOes);GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,oesTex);draw(pOes,stMatrix);EGL14.eglSwapBuffers(egl,window);
    }
    /** x2 AI super resolution: half-size capture, NPU detail, one full-size draw. */
    void superRes(long now,long acquired){
        if(window==EGL14.EGL_NO_SURFACE)return;
        if(sr==0||srW!=capW||srH!=capH){if(!prepareSr())return;}
        long t0=System.nanoTime();
        // 1. Low-res luma, read straight into the NPU's input memory (1 byte per low-res pixel).
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,lumaFbo);GLES20.glViewport(0,0,capW/4,capH);GLES20.glUseProgram(pLuma);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(pLuma,"du"),stMatrix[0]/capW,stMatrix[1]/capW);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,oesTex);draw(pLuma,stMatrix);
        srIn.position(0);GLES20.glReadPixels(0,0,capW/4,capH,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,srIn);
        long t1=System.nanoTime();
        // 2. The same game frame presented again (emulators, paused menus): nothing new to upscale.
        final int step=97,pixels=capW*capH,n=(pixels+step-1)/step;int moved=0;long bright=0;
        if(srLast==null||srLast.length!=n){srLast=new byte[n];srHave=false;}
        for(int k=0,i=0;k<pixels;k+=step,i++){int v=srIn.get(k)&255,d=v-(srLast[i]&255);if(d>DIFF||d<-DIFF)moved++;bright+=v;}
        if(srHave&&moved<n/100&&now-lastTime<STALE_NS){repeatCount++;return;}
        for(int k=0,i=0;k<pixels;k+=step,i++)srLast[i]=srIn.get(k);
        // A black capture (protected content, or a long dark loading screen) only shows a warning: ending the
        // session here would also end it on any game scene that stays black for a few seconds.
        if(bright<n){if(firstBlack==0)firstBlack=now;if(now-firstBlack>4_000_000_000L)srWarning="the capture is black (protected content?)";}else{firstBlack=0;srWarning="";}
        inputCount++;
        // 3. NPU: the detail bilinear upscaling misses, four bytes per low-res pixel.
        double npu;
        try{npu=NativeEngine.srRun(sr);}
        catch(Throwable e){
            if(now-srFailSince>SR_FAIL_WINDOW_NS){srFailSince=now;srFailures=0;}
            android.util.Log.w("FrameLab","NPU pass failed (failure "+(srFailures+1)+" within 30 s), rebuilding the super-resolution graph",e);
            if(++srFailures<=SR_MAX_FAILURES){releaseSr();return;} // the next frame builds the graph again
            srError="AI super resolution stopped after repeated NPU errors: "+String.valueOf(e.getMessage()).split("\n")[0];
            status("FrameLab • "+srError,true);setMode(this,OFF);chooseBackend(OFF);return;
        }
        long t2=System.nanoTime();
        // 4. Detail into a texture, then one full-size pass: bilinear colour + AI detail.
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,resTex);srOut.position(0);
        GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D,0,0,0,capW,capH,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,srOut);
        long t3=System.nanoTime();
        screen();GLES20.glUseProgram(pSr);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(pSr,"t"),0);GLES20.glUniform1i(GLES20.glGetUniformLocation(pSr,"r"),1);
        GLES20.glUniform2f(GLES20.glGetUniformLocation(pSr,"lr"),capW,capH);GLES20.glUniform1f(GLES20.glGetUniformLocation(pSr,"split"),srCompare?0.5f:-1f);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,oesTex);
        draw(pSr,stMatrix);EGL14.eglSwapBuffers(egl,window);
        srHave=true;lastTime=now;realCount++;srCount++;npuSum+=npu;readSum+=(t1-t0)/1e6;texSum+=(acquired-now)/1e6;readCount++;uploadSum+=(t3-t2)/1e6;workSum+=(System.nanoTime()-t0)/1e6;
        stats(now);
    }
    /** Builds (or rebuilds for a new size) the NPU graph and the GL resources of super resolution. */
    boolean prepareSr(){
        releaseSr();
        status(String.format(Locale.US,"FrameLab • SR • preparing the NPU for %dx%d → %dx%d…",capW,capH,2*capW,2*capH),false);
        // Developer override of the model: run-as com.alp.framelab sh -c "echo 1 > files/srmodel" (0 = larger, 1 = faster)
        int model=-1;try{model=Integer.parseInt(new String(java.nio.file.Files.readAllBytes(new File(getFilesDir(),"srmodel").toPath())).trim());}catch(Exception ignored){}
        try{sr=NativeEngine.createSr(capW,capH,getCodeCacheDir().getPath(),SR_BUDGET_MS,model);}
        catch(Throwable e){
            android.util.Log.e("FrameLab","AI super resolution unavailable",e);
            String m=String.valueOf(e.getMessage());srError="AI super resolution unavailable: "+m.split("\n")[0];
            status("FrameLab • "+srError,true);setMode(this,OFF);chooseBackend(OFF);return false;
        }
        srW=capW;srH=capH;srIn=NativeEngine.srInput(sr);srOut=NativeEngine.srOutput(sr);srInfo=NativeEngine.srInfo(sr);
        int[] id=new int[1];
        lumaTex=texture(capW/4,capH,null);GLES20.glGenFramebuffers(1,id,0);lumaFbo=id[0];bind(lumaFbo,lumaTex);
        resTex=texture(capW,capH,null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_NEAREST);GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_NEAREST);
        srHave=false;srLast=null;
        android.util.Log.i("FrameLab","SR "+srInfo);
        return true;
    }
    void releaseSr(){
        if(sr!=0){NativeEngine.destroySr(sr);sr=0;srIn=srOut=null;}
        if(lumaFbo!=0){GLES20.glDeleteFramebuffers(1,new int[]{lumaFbo},0);lumaFbo=0;}
        if(lumaTex!=0){GLES20.glDeleteTextures(1,new int[]{lumaTex},0);lumaTex=0;}
        if(resTex!=0){GLES20.glDeleteTextures(1,new int[]{resTex},0);resTex=0;}
    }
    /** Draws either the newest real frame (pShow) or the generated midpoint (pMid) to the screen. */
    void present(int program){
        if(window==EGL14.EGL_NO_SURFACE)return;
        screen();GLES20.glUseProgram(program);
        if(program==pMid){
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,tex[prev]);GLES20.glUniform1i(GLES20.glGetUniformLocation(program,"t"),0);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,tex[cur]);GLES20.glUniform1i(GLES20.glGetUniformLocation(program,"b"),1);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE2);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,flowTex);GLES20.glUniform1i(GLES20.glGetUniformLocation(program,"fl"),2);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        }else{GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,tex[cur]);}
        draw(program,identity);EGL14.eglSwapBuffers(egl,window);
    }
    void stats(long now){
        if(statsEpoch==0){statsEpoch=now;return;}if(now-statsEpoch<1_000_000_000L)return;
        double seconds=(now-statsEpoch)/1e9;
        final int div=backend==SR?srCount:midCount; // frames the per-frame timings below are averaged over
        String text=backend==OFF?String.format(Locale.US,"FrameLab • OFF • captured %.0f fps",inputCount/seconds)+(srError.isEmpty()?"":" • "+srError)
            :backend==SR?String.format(Locale.US,"FrameLab • SR ×2 AI • %dx%d → %dx%d • captured %.0f/s • shown %.0f/s • NPU %.1f ms • total %.1f ms%s",capW,capH,2*capW,2*capH,inputCount/seconds,realCount/seconds,div>0?npuSum/div:0,div>0?workSum/div:0,srCompare?" • left: bilinear | right: AI":"")+(srWarning.isEmpty()?"":" • "+srWarning)
            :bypassUntil!=0?String.format(Locale.US,"FrameLab • %s • generation paused • captured %.0f fps • retry in %d s",mode,inputCount/seconds,Math.max(0,(bypassUntil-now)/1_000_000_000L)):String.format(Locale.US,"FrameLab • %s • captured %.0f/s • shown real %.0f + generated %.0f/s • motion %.1f ms • total %.1f ms",mode,inputCount/seconds,realCount/seconds,midCount/seconds,midCount>0?npuSum/midCount:0,midCount>0?workSum/midCount:0);
        if(metrics!=null)try{String row=String.format(Locale.US,"%d,%s,%d,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f\n",now,mode,bypassUntil!=0?1:0,inputCount/seconds,realCount/seconds,midCount/seconds,div>0?npuSum/div:0,div>0?flowSum/div:0,div>0?workSum/div:0,readCount>0?readSum/readCount:0,readCount>0?texSum/readCount:0,capW,capH,div>0?uploadSum/div:0);metrics.write(row.getBytes(java.nio.charset.StandardCharsets.UTF_8));metrics.flush();}catch(Exception e){android.util.Log.w("FrameLab","metrics",e);}
        android.util.Log.i("FrameLab","changed-pixel histogram <1,8,24,48,100,200,400,800,1600,3200,6400,more: "+java.util.Arrays.toString(histogram));java.util.Arrays.fill(histogram,0);
        android.util.Log.i("FrameLab",text+" • repeats "+repeatCount+" (avg "+(repeatCount>0?repeatSum/repeatCount:0)+" px) • new avg "+(inputCount>0?newSum/Math.max(1,inputCount):0)+" min "+(newMin==Integer.MAX_VALUE?0:newMin)+" px • scene cuts "+cutCount+" • "+(backend==SR?srInfo.split("\n")[0]+String.format(Locale.US," • readback %.1f ms • upload %.1f ms",readCount>0?readSum/readCount:0,div>0?uploadSum/div:0):device)+String.format(Locale.US," • acquire %.1f ms • priority %s",readCount>0?texSum/readCount:0,highPriority?"high":"normal"));status(text,false);
        realCount=midCount=cutCount=readCount=inputCount=repeatCount=srCount=0;npuSum=flowSum=workSum=readSum=texSum=uploadSum=0;repeatSum=newSum=0;newMin=Integer.MAX_VALUE;statsEpoch=now;
    }
    void draw(int program,float[] matrix){
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(program,"m"),1,false,matrix,0);
        GLES20.glEnableVertexAttribArray(0);GLES20.glVertexAttribPointer(0,2,GLES20.GL_FLOAT,false,0,quad);GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP,0,4);
    }
    int texture(int w,int h,ByteBuffer data){int[] id=new int[1];GLES20.glGenTextures(1,id,0);GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,id[0]);GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D,0,GLES20.GL_RGBA,w,h,0,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,data);params();return id[0];}
    void params(){
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE);GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE);
    }
    void bind(int framebuffer,int texture){
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,framebuffer);GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,GLES20.GL_COLOR_ATTACHMENT0,GLES20.GL_TEXTURE_2D,texture,0);
        if(GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)!=GLES20.GL_FRAMEBUFFER_COMPLETE)throw new IllegalStateException("The GPU framebuffer could not be created");
    }
    int program(String fragment){
        return program("attribute vec2 p;varying vec2 uv;uniform mat4 m;void main(){uv=(m*vec4(p*0.5+0.5,0.0,1.0)).xy;gl_Position=vec4(p,0.0,1.0);}",fragment);
    }
    int program(String vertex,String fragment){
        int program=GLES20.glCreateProgram();GLES20.glAttachShader(program,shader(GLES20.GL_VERTEX_SHADER,vertex));GLES20.glAttachShader(program,shader(GLES20.GL_FRAGMENT_SHADER,fragment));
        GLES20.glBindAttribLocation(program,0,"p");GLES20.glLinkProgram(program);int[] ok=new int[1];GLES20.glGetProgramiv(program,GLES20.GL_LINK_STATUS,ok,0);
        if(ok[0]==0)throw new IllegalStateException("Shader link failed: "+GLES20.glGetProgramInfoLog(program));return program;
    }
    int shader(int type,String source){
        int s=GLES20.glCreateShader(type);GLES20.glShaderSource(s,source);GLES20.glCompileShader(s);int[] ok=new int[1];GLES20.glGetShaderiv(s,GLES20.GL_COMPILE_STATUS,ok,0);
        if(ok[0]==0)throw new IllegalStateException("Shader compile failed: "+GLES20.glGetShaderInfoLog(s));return s;
    }
    void status(String text,boolean failed){OverlayService overlay=OverlayService.instance;if(overlay!=null)overlay.status(text,failed);}
    void error(Throwable e){android.util.Log.e("FrameLab","Capture failed",e);main.post(()->{Toast.makeText(this,"FrameLab: "+e.getMessage(),Toast.LENGTH_LONG).show();stopSelf();});}
    void teardown(){
        worker.removeCallbacksAndMessages(null);
        if(display!=null){display.release();display=null;}
        if(inputSurface!=null){inputSurface.release();inputSurface=null;}
        if(input!=null){input.release();input=null;}
        for(int j=0;j<engines.length;j++)if(engines[j]!=0){NativeEngine.destroyFlow(engines[j]);engines[j]=0;}engine=0;
        if(sr!=0){NativeEngine.destroySr(sr);sr=0;}
        if(metrics!=null)try{metrics.close();}catch(Exception ignored){}
        if(egl!=EGL14.EGL_NO_DISPLAY){
            EGL14.eglMakeCurrent(egl,EGL14.EGL_NO_SURFACE,EGL14.EGL_NO_SURFACE,EGL14.EGL_NO_CONTEXT);
            if(window!=EGL14.EGL_NO_SURFACE){EGL14.eglDestroySurface(egl,window);window=EGL14.EGL_NO_SURFACE;}
            if(context!=EGL14.EGL_NO_CONTEXT)EGL14.eglDestroyContext(egl,context);
            EGL14.eglTerminate(egl);egl=EGL14.EGL_NO_DISPLAY;
        }
        if(projection!=null){projection.stop();projection=null;}
    }
    @Override public void onDestroy(){
        android.util.Log.i("FrameLab","capture service stopping, mode "+mode);
        stopping=true;active=false;
        if(worker!=null){
            // Finish all GL work before the output window disappears.
            CountDownLatch done=new CountDownLatch(1);
            worker.post(()->{try{teardown();}catch(Throwable e){android.util.Log.w("FrameLab","teardown",e);}finally{done.countDown();thread.quitSafely();}});
            try{done.await(1500,TimeUnit.MILLISECONDS);}catch(InterruptedException ignored){}
        }else if(projection!=null)projection.stop();
        OverlayService overlay=OverlayService.instance;if(overlay!=null)overlay.hide();
        stopForeground(STOP_FOREGROUND_REMOVE);super.onDestroy();
    }
}
