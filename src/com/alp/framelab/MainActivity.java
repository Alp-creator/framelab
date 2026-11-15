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
    TextView status; Button start,test; boolean tested=false;volatile boolean busy=false;
    @Override public void onCreate(Bundle state){
        super.onCreate(state);
        ScrollView scroll=new ScrollView(this);LinearLayout root=new LinearLayout(this);root.setOrientation(1);root.setPadding(32,48,32,48);root.setBackgroundColor(Color.rgb(15,23,35));scroll.addView(root);setContentView(scroll);
        label(root,"FrameLab",32,Color.WHITE);label(root,"NPU-assisted frame generation • Experimental 0.3",17,0xff66ddc5);
        label(root,"Finds the motion between two real frames and inserts a generated frame in between. The NPU estimates the motion, the GPU draws the generated frame. Fast motion can cause visual artifacts, and about half a frame of extra display latency is added. Not every game is supported; a higher frame rate is not guaranteed.",17,0xffd4dce8);
        status=label(root,"Turn frame generation on or off. You can also switch it in-game with the MODE button.",16,0xffffd27d);
        Spinner modes=new Spinner(this);modes.setAdapter(new ArrayAdapter<String>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"NPU · frame generation","OFF · capture passthrough only"}));
        modes.setSelection(getPreferences(0).getInt("mode",0)==2?1:0);root.addView(modes);
        modes.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener(){public void onNothingSelected(android.widget.AdapterView<?> p){} public void onItemSelected(android.widget.AdapterView<?> p,View v,int pos,long id){int b=pos==0?0:2;getPreferences(0).edit().putInt("mode",b).apply();CaptureService.setMode(MainActivity.this,b);}});
        test=button(root,"1 · Test the NPU",v->runTest());
        button(root,"Motion comparison demo",v->startActivity(new Intent(this,DemoActivity.class)));
        button(root,"2 · Allow the overlay",v->startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)));
        label(root,"On the Accessibility page that opens, enable the FrameLab Experimental service. It does not read screen content or send touches; it is only used to show the picture on top of the game.",15,0xffaebdce);
        start=button(root,"3 · Try it in a game",v->capture());start.setEnabled(true);
        label(root,"In the sharing dialog choose ‘Share one app’, then pick the game. Do not choose ‘Entire screen’. Open the game in the orientation you want first. To close the overlay, tap STOP in the top right corner or the notification.",16,0xffd4dce8);
        button(root,"Stop",v->{stopService(new Intent(this,CaptureService.class));status.setText("Stopped.");});
        label(root,"Runs entirely on the device. No internet permission, no recording, no audio capture. NPU: motion estimation. GPU: drawing the generated frame. CPU: preparation and the small motion map. Timing measurements are written to a local CSV file; no picture is stored. This version does not use a trained AI model.",14,0xffaebdce);
        if(getIntent().getBooleanExtra("selftest",false))runTest();
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
