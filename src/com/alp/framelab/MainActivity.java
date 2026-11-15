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
        label(root,"FrameLab",32,Color.WHITE);label(root,"NPU destekli ara kare • Deneysel 0.2",17,0xff66ddc5);
        label(root,"İki gerçek kare arasındaki hareketi arar, araya yeni bir görüntü yerleştirir. Görüntü ekran kartında kalır ve tam çözünürlükte gösterilir; NPU yalnızca hareketi hesaplar. Hızlı hareketlerde görüntü hataları ve yaklaşık yarım kare ek gecikme olabilir. Her oyun desteklenmez; FPS artışı garanti edilmez.",17,0xffd4dce8);
        status=label(root,"Önce cihazda NPU testini çalıştır.",16,0xffffd27d);
        test=button(root,"1 · NPU’yu test et",v->runTest());
        button(root,"Hareketli karşılaştırma demosu",v->startActivity(new Intent(this,DemoActivity.class)));
        button(root,"2 · Görüntü katmanına izin ver",v->startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)));
        label(root,"Açılan Erişilebilirlik sayfasında FrameLab Deneysel hizmetini aç. Bu hizmet ekran öğelerini okumaz veya dokunma göndermez; oyun üstünde görüntü göstermek için kullanılır.",15,0xffaebdce);
        start=button(root,"3 · Bir oyunda dene",v->capture());start.setEnabled(false);
        label(root,"Paylaşım penceresinde ‘Tek bir uygulama’yı seç, ardından oyunu seç. ‘Ekranın tamamı’nı seçme. Oyunu önce istediğin yönde aç. Katmanı kapatmak için sağ üstteki DURDUR’a veya bildirime dokun.",16,0xffd4dce8);
        button(root,"Durdur",v->{stopService(new Intent(this,CaptureService.class));status.setText("Durduruldu.");});
        label(root,"Tamamen cihazda çalışır. İnternet izni, kayıt ve ses yakalama yok. NPU: hareket eşleştirme. Ekran kartı: ara karenin çizimi. İşlemci: yalnızca küçük hareket haritasının seçimi. Bu sürüm eğitilmiş bir yapay zekâ modeli kullanmaz.",14,0xffaebdce);
        if(getIntent().getBooleanExtra("selftest",false))runTest();
    }
    TextView label(LinearLayout root,String text,int sp,int color){TextView t=new TextView(this);t.setText(text);t.setTextSize(sp);t.setTextColor(color);t.setPadding(0,10,0,16);root.addView(t);return t;}
    Button button(LinearLayout root,String text,View.OnClickListener listener){Button b=new Button(this);b.setText(text);b.setAllCaps(false);b.setOnClickListener(listener);root.addView(b,new LinearLayout.LayoutParams(-1,-2));return b;}
    void runTest(){
        if(busy)return;busy=true;test.setEnabled(false);status.setText("Hızlandırıcıda gerçek hesap ve doğruluk testi çalışıyor…");
        new Thread(()->{long h=0;String report;boolean ok=false;
            try{h=NativeEngine.create();report=NativeEngine.test(h);ok=true;}catch(Throwable e){report="Test başarısız: "+e.getMessage();}finally{if(h!=0)NativeEngine.destroy(h);}
            android.util.Log.i("FrameLab",report);
            try(FileOutputStream f=openFileOutput("npu-test.txt",MODE_PRIVATE)){f.write(report.getBytes(StandardCharsets.UTF_8));}catch(Exception ignored){}
            final boolean passed=ok;final String result=report;
            runOnUiThread(()->{busy=false;tested=passed;if(isDestroyed())return;test.setEnabled(true);start.setEnabled(passed);status.setText((passed?"NPU sürücüsü testi geçti.\n":"")+result);});
        },"NPU-test").start();
    }
    void capture(){
        if(!tested)return;
        if(OverlayService.instance==null){new AlertDialog.Builder(this).setMessage("Önce 2. adımda FrameLab görüntü katmanı hizmetini aç.").setPositiveButton("Tamam",null).show();return;}
        if(CaptureService.active){Toast.makeText(this,"Önce mevcut oturumu durdur.",Toast.LENGTH_LONG).show();return;}
        new AlertDialog.Builder(this).setTitle("Yalnızca tek uygulama paylaşımı")
            .setMessage("Bir sonraki Android penceresinde ‘Tek bir uygulama’yı ve ardından oyunu seç. Tüm ekran paylaşımı görüntü döngüsüne veya siyah ekrana yol açabilir. Ekran kaydedilmez.")
            .setPositiveButton("Oyunu seç",(d,w)->startActivityForResult(getSystemService(MediaProjectionManager.class).createScreenCaptureIntent(),42)).setNegativeButton("İptal",null).show();
    }
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request==42&&result==RESULT_OK&&data!=null){Intent i=new Intent(this,CaptureService.class).putExtra("result",result).putExtra("token",data);startForegroundService(i);moveTaskToBack(true);}}
}
