package com.alp.framelab;

import android.app.*;
import android.content.*;
import android.graphics.Color;
import android.media.projection.MediaProjectionManager;
import android.os.*;
import android.provider.Settings;
import android.view.*;
import android.widget.*;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {
    TextView status; Button start,test,srTest; boolean tested=false;volatile boolean busy=false;
    @Override public void onCreate(Bundle state){
        super.onCreate(state);
        ScrollView scroll=new ScrollView(this);LinearLayout root=new LinearLayout(this);root.setOrientation(1);root.setPadding(32,48,32,48);root.setBackgroundColor(Color.rgb(15,23,35));scroll.addView(root);setContentView(scroll);
        label(root,"FrameLab",32,Color.WHITE);label(root,"NPU frame generation and AI super resolution • Experimental 0.4",17,0xff66ddc5);
        label(root,"NPU: finds the motion between two real frames and inserts a generated frame in between. Fast motion can cause visual artifacts, and about half a frame of extra display latency is added.\n\nSR: captures the game at half resolution and lets a small trained neural network on the NPU restore the full-resolution detail (x2, up to 4K output). Works best when the game itself renders at a lower resolution. Not every game is supported.",17,0xffd4dce8);
        status=label(root,"Choose a mode. You can also switch it in-game with the MODE button (long press in SR: split-screen comparison).",16,0xffffd27d);
        Spinner modes=new Spinner(this);modes.setAdapter(new ArrayAdapter<String>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"NPU · frame generation","SR · AI super resolution x2 (NPU)","OFF · capture passthrough only"}));
        modes.setSelection(Math.max(0,Math.min(2,getPreferences(0).getInt("mode",0))));root.addView(modes);
        modes.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener(){public void onNothingSelected(android.widget.AdapterView<?> p){} public void onItemSelected(android.widget.AdapterView<?> p,View v,int pos,long id){getPreferences(0).edit().putInt("mode",pos).apply();CaptureService.setMode(MainActivity.this,pos);}});
        test=button(root,"1 · Test the NPU",v->runTest());
        srTest=button(root,"Test AI super resolution on the NPU",v->runSrTest());
        button(root,"Motion comparison demo",v->startActivity(new Intent(this,DemoActivity.class)));
        button(root,"2 · Allow the overlay",v->startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)));
        label(root,"On the Accessibility page that opens, enable the FrameLab Experimental service. It does not read screen content or send touches; it is only used to show the picture on top of the game.",15,0xffaebdce);
        start=button(root,"3 · Try it in a game",v->capture());start.setEnabled(true);
        label(root,"In the sharing dialog choose ‘Share one app’, then pick the game. Do not choose ‘Entire screen’. Open the game in the orientation you want first. To close the overlay, tap STOP in the top right corner or the notification.",16,0xffd4dce8);
        button(root,"Stop",v->{stopService(new Intent(this,CaptureService.class));status.setText("Stopped.");});
        label(root,"Runs entirely on the device. No internet permission, no recording, no audio capture. Frame generation: the NPU estimates motion (analytic block matching, not a trained model), the GPU draws the generated frame. Super resolution: a trained 5k-parameter network runs on the NPU; colour is upscaled on the GPU. Timing measurements are written to a local CSV file; no picture is stored.",14,0xffaebdce);
        if(getIntent().getBooleanExtra("selftest",false))runTest();
        if(getIntent().getBooleanExtra("srtest",false))runSrTest();
    }
    /** Builds the super-resolution graph for this screen (half size in, full size out) and times it, plus 4K. */
    void runSrTest(){
        if(busy)return;busy=true;srTest.setEnabled(false);status.setText("Compiling and checking the super-resolution network on the accelerator…");
        android.graphics.Rect b=getSystemService(WindowManager.class).getMaximumWindowMetrics().getBounds();
        int w=Math.max(b.width(),b.height()),h=Math.min(b.width(),b.height()); // games run in landscape
        new Thread(()->{String report;
            try{report=NativeEngine.srTest(w/2/4*4,h/2/2*2,getCodeCacheDir().getPath());}catch(Throwable e){report="Super-resolution test failed: "+e.getMessage();}
            android.util.Log.i("FrameLab",report);
            try(FileOutputStream f=openFileOutput("sr-test.txt",MODE_PRIVATE)){f.write(report.getBytes(StandardCharsets.UTF_8));}catch(Exception ignored){}
            final String result=report;
            runOnUiThread(()->{busy=false;if(isDestroyed())return;srTest.setEnabled(true);status.setText(result);});
        },"SR-test").start();
    }
    TextView label(LinearLayout root,String text,int sp,int color){TextView t=new TextView(this);t.setText(text);t.setTextSize(sp);t.setTextColor(color);t.setPadding(0,10,0,16);root.addView(t);return t;}
    Button button(LinearLayout root,String text,View.OnClickListener listener){Button b=new Button(this);b.setText(text);b.setAllCaps(false);b.setOnClickListener(listener);root.addView(b,new LinearLayout.LayoutParams(-1,-2));return b;}
    void runTest(){
        if(busy)return;busy=true;test.setEnabled(false);status.setText("Running a real computation and accuracy test on the accelerator…");
        new Thread(()->{long h=0;String report;boolean ok=false;
            try{h=NativeEngine.create();report=NativeEngine.test(h);ok=true;}catch(Throwable e){report="Test failed: "+e.getMessage();}finally{if(h!=0)NativeEngine.destroy(h);}
            android.util.Log.i("FrameLab",report);
            try(FileOutputStream f=openFileOutput("npu-test.txt",MODE_PRIVATE)){f.write(report.getBytes(StandardCharsets.UTF_8));}catch(Exception ignored){}
            final boolean passed=ok;final String result=report;
            runOnUiThread(()->{busy=false;tested=passed;if(isDestroyed())return;test.setEnabled(true);start.setEnabled(true);status.setText((passed?"NPU driver test passed.\n":"")+result);});
        },"NPU-test").start();
    }
    void capture(){

        if(OverlayService.instance==null){new AlertDialog.Builder(this).setMessage("Enable the FrameLab overlay service in step 2 first.").setPositiveButton("OK",null).show();return;}
        if(CaptureService.active){Toast.makeText(this,"Stop the current session first.",Toast.LENGTH_LONG).show();return;}
        new AlertDialog.Builder(this).setTitle("Single-app sharing only")
            .setMessage("In the next Android dialog choose ‘Share one app’ and then the game. Sharing the entire screen can cause a picture loop or a black screen. Nothing is recorded.")
            .setPositiveButton("Choose the game",(d,w)->startActivityForResult(getSystemService(MediaProjectionManager.class).createScreenCaptureIntent(),42)).setNegativeButton("Cancel",null).show();
    }
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request==42&&result==RESULT_OK&&data!=null){Intent i=new Intent(this,CaptureService.class).putExtra("result",result).putExtra("token",data);startForegroundService(i);moveTaskToBack(true);}}
}
