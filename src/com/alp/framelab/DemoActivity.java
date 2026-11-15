package com.alp.framelab;
import android.app.Activity;
import android.graphics.*;
import android.os.*;
import android.view.*;
import android.widget.*;
import java.util.Locale;

public class DemoActivity extends Activity {
    HandlerThread thread;Handler worker;volatile boolean closed=false;long engine;FrameView source,result;
    @Override public void onCreate(Bundle saved){
        super.onCreate(saved);getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LinearLayout root=new LinearLayout(this);root.setOrientation(1);root.setPadding(12,28,12,20);root.setBackgroundColor(0xff101823);
        TextView title=new TextView(this);title.setText("Motion demo · left: 20 real fps · right: target 20 real + 20 generated fps");title.setTextSize(19);title.setTextColor(Color.WHITE);root.addView(title);
        LinearLayout pair=new LinearLayout(this);source=new FrameView(this);result=new FrameView(this);pair.addView(source,new LinearLayout.LayoutParams(0,-1,1));pair.addView(result,new LinearLayout.LayoutParams(0,-1,1));root.addView(pair,new LinearLayout.LayoutParams(-1,0,1));
        Button back=new Button(this);back.setText("Back");back.setOnClickListener(v->finish());root.addView(back);setContentView(root);
        thread=new HandlerThread("FrameLab-demo");thread.start();worker=new Handler(thread.getLooper());worker.post(()->runDemo());
    }
    void runDemo(){try{
        engine=NativeEngine.create();String test=NativeEngine.test(engine);android.util.Log.i("FrameLab","Demo: "+test);
        final int w=480,h=288;int[] old=null;long oldTime=0,start=System.nanoTime(),next=start;int index=0;
        while(!closed){
            long now=System.nanoTime();if(now<next){try{Thread.sleep(Math.max(1,(next-now)/1_000_000));}catch(InterruptedException e){break;}continue;}
            next=Math.max(next+50_000_000L,now);int[] current=scene(w,h,index++);Bitmap real=Bitmap.createBitmap(current,w,h,Bitmap.Config.ARGB_8888);
            source.performance="Source • same 100 ms buffer";source.offer(real,now,false);
            if(old!=null){int[] mid=new int[w*h];long t=System.nanoTime();double npu=NativeEngine.generate(engine,old,current,mid,w,h);double total=(System.nanoTime()-t)/1e6;result.performance=String.format(Locale.US,"NPU %.1f ms • total %.1f ms",npu,total);result.offer(Bitmap.createBitmap(mid,w,h,Bitmap.Config.ARGB_8888),(oldTime+now)/2,true);}
            result.offer(real,now,false);old=current;oldTime=now;
        }
    }catch(Throwable e){android.util.Log.e("FrameLab","Demo failed",e);runOnUiThread(()->{result.failed=true;result.performance="Hata: "+e.getMessage();result.invalidate();});}
    finally{if(engine!=0){NativeEngine.destroy(engine);engine=0;}thread.quitSafely();}}
    static int[] scene(int w,int h,int frame){
        int[] p=new int[w*h];for(int y=0;y<h;y++)for(int x=0;x<w;x++){int sx=x+frame*6;int r=30+((sx/18+y/18)%2)*45,g=60+(sx%120)/2,b=95+(y%100);p[y*w+x]=0xff000000|(r<<16)|(g<<8)|b;}
        // Separately moving textured subject, to expose block and occlusion errors.
        int center=80+(frame*3)%(w-160);
        for(int y=90;y<200;y++)for(int x=Math.max(0,center);x<Math.min(w,center+80);x++){int stripe=((x-center)/8+y/8)%2;p[y*w+x]=stripe==0?0xffffca65:0xffd27332;}
        return p;
    }
    @Override public void onDestroy(){closed=true;if(thread!=null)thread.interrupt();super.onDestroy();}
}
