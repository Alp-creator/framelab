package com.alp.framelab;
import android.accessibilityservice.AccessibilityService;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.view.*;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TextView;

public class OverlayService extends AccessibilityService {
    static volatile OverlayService instance;
    FrameLayout root; SurfaceView surface; TextView hud; Button stop,mode; WindowManager wm;
    @Override protected void onServiceConnected(){super.onServiceConnected();instance=this;wm=getSystemService(WindowManager.class);}
    /** Full-screen output window. Touches pass through to the game underneath. */
    SurfaceView show(boolean secure){
        hide();
        root=new FrameLayout(this);root.setBackgroundColor(Color.BLACK);
        surface=new SurfaceView(this);root.addView(surface,new FrameLayout.LayoutParams(-1,-1));
        float d=getResources().getDisplayMetrics().density;
        hud=new TextView(this);hud.setTextSize(11);hud.setTextColor(0xff74f0d0);hud.setBackgroundColor(0x99000000);hud.setPadding((int)(8*d),(int)(3*d),(int)(8*d),(int)(3*d));hud.setText("FrameLab is starting…");
        root.addView(hud,new FrameLayout.LayoutParams(-2,-2,Gravity.TOP|Gravity.START));
        int flags=WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE|WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE|WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN|WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED|WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON;
        // Secure keeps the output out of any capture, so a wrong "entire screen" choice cannot loop.
        if(secure)flags|=WindowManager.LayoutParams.FLAG_SECURE;
        WindowManager.LayoutParams p=new WindowManager.LayoutParams(-1,-1,WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,flags,PixelFormat.OPAQUE);
        p.layoutInDisplayCutoutMode=WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;p.setFitInsetsTypes(0);p.setTitle("FrameLab output");
        wm.addView(root,p);
        stop=new Button(this);stop.setText("STOP");stop.setAllCaps(false);stop.setAlpha(.55f);stop.setOnClickListener(v->{stopService(new Intent(this,CaptureService.class));hide();});
        WindowManager.LayoutParams s=new WindowManager.LayoutParams((int)(96*d),(int)(44*d),WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,PixelFormat.TRANSLUCENT);
        s.gravity=Gravity.TOP|Gravity.END;s.setTitle("FrameLab stop");wm.addView(stop,s);
        // Tap: frame generation -> AI super resolution -> off. Long press in super resolution: split-screen comparison.
        mode=new Button(this);mode.setText("MODE");mode.setAlpha(.55f);mode.setOnClickListener(v->{CaptureService.setMode(this,(CaptureService.requestedBackend+1)%3);});
        mode.setOnLongClickListener(v->{CaptureService.srCompare=!CaptureService.srCompare;return true;});
        WindowManager.LayoutParams m=new WindowManager.LayoutParams((int)(80*d),(int)(44*d),WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,PixelFormat.TRANSLUCENT);m.gravity=Gravity.TOP|Gravity.END;m.x=(int)(96*d);m.setTitle("FrameLab mode");wm.addView(mode,m);return surface;
    }
    void status(String text,boolean error){TextView h=hud;if(h!=null)h.post(()->{h.setText(text);h.setTextColor(error?0xffff8d8d:0xff74f0d0);});}
    // A view may never have been attached (for example when the service token was not valid yet), so removal must not throw.
    void remove(View v){if(v!=null&&wm!=null)try{wm.removeViewImmediate(v);}catch(RuntimeException ignored){}}
    void hide(){remove(root);root=null;surface=null;hud=null;remove(stop);stop=null;remove(mode);mode=null;}
    @Override public void onAccessibilityEvent(AccessibilityEvent event){} // No events requested, no UI inspection.
    @Override public void onInterrupt(){stopService(new Intent(this,CaptureService.class));hide();}
    @Override public void onDestroy(){stopService(new Intent(this,CaptureService.class));hide();instance=null;super.onDestroy();}
}
