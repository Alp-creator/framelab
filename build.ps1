param(
    [string]$AndroidRoot='C:/Program Files/Unity/Hub/Editor/6000.6.2f1/Editor/Data/PlaybackEngines/AndroidPlayer',
    [string]$JavaRoot='C:/Program Files/Android/Android Studio2/jbr'
)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
function Checked([scriptblock]$Action) { & $Action; if($LASTEXITCODE -ne 0){throw "Build step failed: $LASTEXITCODE"} }
$sdk=Join-Path $AndroidRoot 'SDK'
$ndk=Join-Path $AndroidRoot 'NDK'
$bt=Join-Path $sdk 'build-tools/36.0.0'
$androidJar=Join-Path $sdk 'platforms/android-36/android.jar'
$env:JAVA_HOME=$JavaRoot
$env:PATH="$JavaRoot/bin;$env:PATH"
New-Item -ItemType Directory -Force -Path build,build/classes,build/generated,build/dex,build/lib/arm64-v8a | Out-Null
Checked { & "$ndk/toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android34-clang++.cmd" -std=c++17 -O3 -fPIC -shared -static-libstdc++ '-Wl,-z,max-page-size=16384' native/bridge.cpp -lneuralnetworks -o build/lib/arm64-v8a/libframelab.so }
Checked { & "$bt/aapt2.exe" compile --dir res -o build/resources.zip }
Checked { & "$bt/aapt2.exe" link -o build/base.apk --manifest AndroidManifest.xml -I $androidJar --java build/generated build/resources.zip }
$sources=@(Get-ChildItem src,build/generated -Recurse -Filter '*.java' | ForEach-Object {$_.FullName})
Checked { & "$JavaRoot/bin/javac.exe" -encoding UTF-8 -source 17 -target 17 -classpath $androidJar -d build/classes @sources }
$classes=@(Get-ChildItem build/classes -Recurse -Filter '*.class' | ForEach-Object {$_.FullName})
Checked { & "$bt/d8.bat" --lib $androidJar --min-api 34 --output build/dex @classes }
Copy-Item -LiteralPath build/base.apk -Destination build/unsigned.apk -Force
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::Open((Join-Path $PSScriptRoot 'build/unsigned.apk'),[IO.Compression.ZipArchiveMode]::Update)
try{
    [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,(Join-Path $PSScriptRoot 'build/dex/classes.dex'),'classes.dex') | Out-Null
    [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,(Join-Path $PSScriptRoot 'build/lib/arm64-v8a/libframelab.so'),'lib/arm64-v8a/libframelab.so') | Out-Null
}finally{$zip.Dispose()}
Checked { & "$bt/zipalign.exe" -f -P 16 4 build/unsigned.apk build/aligned.apk }
if(!(Test-Path -LiteralPath build/development.jks)){
    Checked { & "$JavaRoot/bin/keytool.exe" -genkeypair -keystore build/development.jks -storepass android -keypass android -alias framelab -keyalg RSA -keysize 2048 -validity 3650 -dname 'CN=FrameLab Local Development' }
}
Checked { & "$bt/apksigner.bat" sign --ks build/development.jks --ks-key-alias framelab --ks-pass pass:android --key-pass pass:android --out build/FrameLab-0.2.apk build/aligned.apk }
Checked { & "$bt/apksigner.bat" verify --verbose build/FrameLab-0.2.apk }
Write-Output "APK ready: $PSScriptRoot/build/FrameLab-0.2.apk"
