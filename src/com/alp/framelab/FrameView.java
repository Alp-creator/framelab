package com.alp.framelab;
import android.content.Context;
import android.graphics.*;
import android.view.*;
import java.util.ArrayDeque;
import java.util.Locale;

final class FrameView extends View implements Choreographer.FrameCallback {
    static final long DELAY_NS=100_000_000L;
    static class Frame {final Bitmap image;final long due;final boolean generated;Frame(Bitmap b,long t,boolean g){image=b;due=t;generated=g;}}
    final ArrayDeque<Frame> queue=new ArrayDeque<>();final Paint paint=new Paint(Paint.FILTER_BITMAP_FLAG);final RectF dest=new RectF();
    Bitmap displayed; boolean running=false, generated=false;int realCount=0,genCount=0,dropped=0;long epoch=System.nanoTime();
    volatile String performance="Başlatılıyor…";String rates="";volatile boolean failed=false;
    FrameView(Context c){super(c);setBackgroundColor(Color.BLACK);}
    synchronized void offer(Bitmap image,long sourceTime,boolean mid){
        if(!running)return;
        while(queue.size()>=12){queue.removeFirst();dropped++;}
        queue.addLast(new Frame(image,sourceTime+DELAY_NS,mid));
    }
    synchronized void clearFrames(){queue.clear();post(()->{displayed=null;invalidate();});}
    @Override protected void onAttachedToWindow(){super.onAttachedToWindow();synchronized(this){running=true;}Choreographer.getInstance().postFrameCallback(this);}
    @Override protected void onDetachedFromWindow(){synchronized(this){running=false;queue.clear();displayed=null;}Choreographer.getInstance().removeFrameCallback(this);super.onDetachedFromWindow();}
    @Override public void doFrame(long t){
        if(!running)return;
        Frame frame=null;long now=System.nanoTime();
        synchronized(this){while(!queue.isEmpty()&&queue.peekFirst().due<=now){if(frame!=null)dropped++;frame=queue.removeFirst();}}
        if(frame!=null){displayed=frame.image;generated=frame.generated;if(generated)genCount++;else realCount++;invalidate();}
        if(now-epoch>=1_000_000_000L){double seconds=(now-epoch)/1e9;rates=String.format(Locale.US,"Gösterilen gerçek %.1f/s + ara %.1f/s  |  atlanan %d",realCount/seconds,genCount/seconds,dropped);realCount=genCount=dropped=0;epoch=now;invalidate();}
        Choreographer.getInstance().postFrameCallback(this);
    }
    @Override protected void onDraw(Canvas c){
        super.onDraw(c);
        if(displayed!=null){float scale=Math.min((float)getWidth()/displayed.getWidth(),(float)getHeight()/displayed.getHeight());float w=displayed.getWidth()*scale,h=displayed.getHeight()*scale;dest.set((getWidth()-w)/2,(getHeight()-h)/2,(getWidth()+w)/2,(getHeight()+h)/2);paint.setColor(Color.WHITE);c.drawBitmap(displayed,null,dest,paint);}
        float d=getResources().getDisplayMetrics().density;paint.setColor(0xc8000000);c.drawRect(0,0,getWidth(),62*d,paint);paint.setTextSize(12*d);paint.setColor(failed?0xffff8d8d:0xff74f0d0);
        c.drawText("FrameLab DENEYSEL • "+performance,10*d,21*d,paint);paint.setColor(Color.WHITE);c.drawText(rates,10*d,43*d,paint);
    }
}
