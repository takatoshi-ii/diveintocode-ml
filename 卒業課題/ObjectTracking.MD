# Signateの第3回AIエッジコンテストをPolyYOLOを使ってやってみた


Dive Into Codeの卒業課題に、Signateでの物体追跡のコンペを選択したので、  
２位に入賞されたIRAFM-AIチームが採用していたPolyYOLOについてWikiにチュートリアルが詳細にまとめられていたので実際にやってみた。  

元記事は以下のWikiに丁寧に分かりやすくまとめられているので(英語だけど)そちらを参考にして個人的にまとめてみます。   
(参照)[https://gitlab.com/irafm-ai/signate_3rd_ai_edge_competition/-/tree/master](https://gitlab.com/irafm-ai/signate_3rd_ai_edge_competition/-/tree/master)


本記事の内容を簡単にスライドにまとめているので、参考にして頂ければと思います。  
[https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/DIC_2020%E5%B9%B4_4%E6%9C%88%E6%9C%9F_%E5%8D%92%E6%A5%AD%E7%99%BA%E8%A1%A8.pdf](https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/DIC_2020%E5%B9%B4_4%E6%9C%88%E6%9C%9F_%E5%8D%92%E6%A5%AD%E7%99%BA%E8%A1%A8.pdf)

## 学習/評価環境
ローカルPC  
OS:Windows10 Home  
GPU:NVIDIA Geforce GTX 1060  
python==3.7.6
FrameWork:tensorflow-gpu==1.15.0  
keras==2.2.5


※Google Colab にGoogle Driveを接続してもやりましたが、パスの”My Drive”に半角スペースが含まれる為、
作成するアノテーションファイルを読み出す際にSplitが面倒になる為（各バウンディングボックスの区切りが半角スペースなので）
ローカルの非力なGPUで実行しました。

## 学習/評価フロー
PolyYOLOを用いたフローは以下の様になっています。  
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/flow.png">

①入力動画をフレーム単位で静止画に切り出し  
②複数のPloy-YOLO検出器で学習(PolyYOLO a ～ e まで５つの検出器の学習が必要)  
③WBF(Weighted Box Fusion)でアンサンブル学習  
④Refiner:検出境界を高精度化  
⑤クラス分類(自動車、歩行者 分類)  
⑥フレーム間オブジェクトの関連付け(追跡処理)  

# 1⃣ 学習
　学習用のコードは、  
[https://github.com/takatoshi-ii/diveintocode-ml/tree/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/signate_3rd_ai_edge_competition/source_codes](https://github.com/takatoshi-ii/diveintocode-ml/tree/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/signate_3rd_ai_edge_competition/source_codes)  
にすべて格納しています。

# 1 Poly-YOLOトレーニング

## 1.1 データの準備
### 1.1.1 入力動画をフレーム単位で静止画に切り出す。  
　 動画当たり、２分（１２０秒）/５fpsなので６００フレームの画像が取得できます。  
 　提供されている動画数が２５個なので合計１５,０００個の静止画が取得できる事になります。  
 　ただ、物体検知を学習するのに１５,０００個では少なすぎるので、以下のデータ拡張を行います。  
 　その中でtrain_00.mp4の動画をvalidation用に使います。
 <br>

 ＜コード概要＞  
 prepare_data_decode_movies.ipynbを使用すします。  
 2つめのセルに以下の情報を設定  
 path_to_videos：train_xx.mp4があるフォルダを指定します  
 path_to_save_imgs：切り出したフレーム画像を保存するフォルダを指定します

 ```python
 #define paths
 path_to_videos    = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\train_videos' #where are videos
 path_to_save_imgs = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images' #where we save videos
 ```

セル１～３を順次実行すると指定したフォルダに動画毎のフォルダが作成され、フレーム画像が作成されます。


### 1.1.2 JSONラベルをPolyYOLOラベル形式に変換する  
　　　画像パス␣x1,y1,x2,y2,classNo␣x1,y1,x2,y2,classNo…  
　　　（␣：半角スペース）  
     例）D:/SIGNATE/Signate_3rd_AI_edge_competition/images/train_01/0.jpg 896,480,1020,591,0 1046,468,1108,526,0 ...  
<br>
     この処理によってアノテーションデータdata_for_yolo_training.txtが指定したフォルダに作成されます。   
     検証データ用にdata_for_yolo_validating.txtを作成し、data_for_yolo_training.txtから
     /train_00/～.jpgに該当する行を切り取り、そこに張り付けます。

 ＜コード概要＞  
prepare_data_prepare_labels.ipynbを使用します。  
２つ目のセルに科各種パスを指定します。

path_labels：Signateから提供されているアノテーションデータを格納しているホルダを指定します。  
path_images：1.1.1で切り出したフレーム画像のパスを指定します。  
path_out_file：変換後のアノテーションファイルの出力先を指定します。  

```python
#define paths
path_labels    = r'D:/SIGNATE/Signate_3rd_AI_edge_competition/train_annotations'      
path_images    = r'D:/SIGNATE/Signate_3rd_AI_edge_competition/images' #path where, the are the decoded train images
path_out_file  = r'D:/SIGNATE/Signate_3rd_AI_edge_competition/data_for_yolo_training.txt' #file used for training models
```

3つめのセルに検出したいクラスを指定します。
今回のコンペの課題は、乗用車と歩行者のみです。  
car：乗用車  
Pedestrian：歩行者  

```python
#classes = ['Car', 'Pedestrian', 'Truck', 'Signal', 'Signs', 'Bicycle', 'Motorbike', 'Bus', 'Svehicle', 'Train']
classes = ['Car', 'Pedestrian']
```

Signateから提供されているアノテーションデータには信号機(Signal)やトラック、バス、自転車などのクラスも  
アノテーションラベルが付与されています。  
（乗用車とトラック、バス等は別クラスの扱いです。）  
ここのclassesに指定してやれば、検知可能となります。  
例えば、信号機を検知して、そのカラーヒストグラムを判断し、青信号/赤信号等を検知するといった事も可能と考えられます。  
 今回のコンペの課題からそれるので対応は見送ります。
（より学習に時間が必要になると考えられます。）

セルの１～４まで実行すると、アノテーションファイルdata_for_yolo_training.txtが作成されます。

 先述の通り、data_for_yolo_training.txtからtrain_00/~.jpgに該当する行を切り取り
data_for_yolo_validating.txtに貼り付けます。

今後data_for_yolo_training.txt及びdata_for_yolo_validating.txtに行を追加していくことになります。


### 1.1.3 歩行者のみを使用してPolyYOLOラベル形式を作成する  
車のクラスと歩行者のクラスの間に不均衡があるため、歩行者のみをピックアップしたアノテーションデータを作成します。  
このアノテーションデータは、PolyYOLO＃4モデルのトレーニングにのみ個別に使用されます。

＜コード概要＞
prepare_data_prepare_labels.ipynbを使用します。  
1.1.2と同じノートブックを使用するため、必要に応じて別名で保存してコピーを作成するなどの対応をして下さい。  
1.1.2と同様に２番目のセルに各種パスを設定して下さい。

path_labels：Signateから提供されているアノテーションデータを格納しているホルダを指定します。  
path_images：1.1.1で切り出したフレーム画像のパスを指定します。  
path_out_file：変換後のアノテーションファイルの出力先を指定します。

```python
#define paths
path_labels    = r'D:/SIGNATE/Signate_3rd_AI_edge_competition/train_annotations'      
path_images    = r'D:/SIGNATE/Signate_3rd_AI_edge_competition/images' #path where, the are the decoded train images
path_out_file  = r'D:/SIGNATE/Signate_3rd_AI_edge_competition/data_for_yolo_training_pedestrians.txt' #file used for training models
```

※path_out_fileは1.1.2とは別のパスにしておいてください。

3つめのセルに検出したいクラスを指定します。
PolyYOLO＃4モデルでは<u>歩行者のみ</u>のクラス分けにします。  
Pedestrian：歩行者  

```python
#classes = ['Car', 'Pedestrian', 'Truck', 'Signal', 'Signs', 'Bicycle', 'Motorbike', 'Bus', 'Svehicle', 'Train']
#classes = ['Car', 'Pedestrian']
classes = ['Pedestrian']
```

セルの１～４まで実行すると、アノテーションファイルdata_for_yolo_training_pedestrians.txtが作成されます。


1.1.2と同様に、検証データ用にdata_for_yolo_validating_pedestrians.txtを作成し、data_for_yolo_training_pedestrians.txtから
/train_00/～.jpgに該当する行を切り取り、そこに張り付けます。

※ここで作成したdata_for_yolo_training_pedestrians.txt及び、data_for_yolo_validating_pedestrians.txtは、  
PolyYOLO＃4モデルのトレーニング時にのみ使用するので、data_for_yolo_training.txt及びdata_for_yolo_validating.txtに追記はしないでください。



## 1.1.4 インペイントを生成する  
### ⅰ）インペイントを作成  
ランダムで２つの画像を選択し、一方をベースとして、もう片方の画像のバウンディングボックスの情報を移植します。
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/124_resize.jpg" width="80%" height="80%">  
本来ありえない場所に他のバウンディングボックスが移植されているのが分かると思います。  
この処理を１００００画像に対して行います。  

＜コード概要＞
inpaint_data.ipynbを使用します。  
最初のコードセルで入出力のパスを指定します。

labels_in：1.1.2で作成したpolyYOLOラベルのアノテーションファイルを指定します。  
labels_out：出力するinpaintアノテーションファイルを指定します。  
path_out_file：インペイント画像ファイルの出力先を指定します。

```python
labels_in       = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
labels_out      = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training_inpaint.txt'
out_dir         = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\imags\inpaint/' #the dir will be created
```

out_dirはフォルダを指定するため、末尾にスラッシュが必須です。  

このセルを実行すると、out_dirで指定したフォルダにインペイント画像が、labels_outで指定したパスにアノテーションファイルが作成されます。  
ここで作成されたdata_for_yolo_training_inpaint.txtの内容をdata_for_yolo_training.txtに追記します。  
（元のデータは削除しないように注意します）


### ⅱ）インペイントクロップを作成  
続いてインペイントクロップデータを作成します。  
作成方法はⅰ）とほぼ同様ですが、ランダム抽出した２つの画像の内、移植元画像のバウンディングボックスを以下の様に変形させます。  

乗用車の場合：移植元バウンディングボックスを高さ方向に６０～９０％でカットします。  
歩行者の場合：移植元バウンディングボックスを高さ方向に３０～９０％でカットします。  

このように変形させたバウンディングボックスを移植先に貼り付け新しいデータを作成します。
この処理も同様に１００００画像に対して行います。

＜コード概要＞  
ⅰ）と同様にinpaint_data.ipynbを使用します。  
２つ目のコードセルで入出力のパスを指定します。

labels_in：1.1.2で作成したpolyYOLOラベルのアノテーションファイルを指定します。  
labels_out：出力するinpaint_cropアノテーションファイルを指定します。  
out_file：インペイントクロップ画像ファイルの出力先を指定します。  

```python
labels_in       = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
labels_out      = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training_inpaint_crop.txt'
out_dir         = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\imags\inpaint_crop/'  #the dir will be created
```

out_dirはフォルダを指定するため、末尾にスラッシュが必須です。  

このセルを実行すると、out_dirで指定したフォルダにインペイントクロップ画像が、labels_outで指定したパスにアノテーションファイルが作成されます。  
ここで作成されたdata_for_yolo_training_inpaint_crop.txtの内容をdata_for_yolo_training.txtに追記します。  
（元のデータは削除しないように注意します）



## 1.1.5モザイクを生成する  
ランダムで２つの画像を選択し、500ピクセル≦ｘ座標≦(1936-500)ピクセルのランダムな位置で縦方向に切断し、2つの画像を接合し1つの画像を作成する。  
（対象は1.1.4で作成したインペイント画像も含まれます）
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/mosaic33.jpg" width="80%" height="80%">  
この処理も同様に１００００画像に対して行います。

＜コード概要＞
mosaic_data.ipynbを使用します。  
最初のコードセルで入出力のパスを指定します。

labels_in：1.1.4で作成したインペイントも含めたアノテーションファイルを指定します。  
labels_out：出力するモザイクアノテーションファイルを指定します。  
out_file：モザイク画像ファイルの出力先を指定します。

```python
labels_in       = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
labels_out      = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training_mosaic.txt'
out_dir         = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\imags\mosaic/'
```

out_dirはフォルダを指定するため、末尾にスラッシュが必須です。

ノートブック全体を実行します。（非常に長い時間がかかります。）

終了するとdata_for_yolo_training_mosaic.txtファイルが作成されているので、同様に中身をdata_for_yolo_training.txtに追記します。  
（元のデータは削除しないように注意します）


以上でPolyYOLOトレーニングの為のデータ拡張は終了です。  
ここまでで、data_for_yolo_training.txtには約44400のアノテーションデータが格納されているはずです。



## 1.2 Poly-YOLO ネットワークトレーニング
PolyYOLOネットワークではそれぞれ以下の解像度でトレーニングを行います。  
\#1) 448×864(自動車、歩行者)  
\#2) 352×704(自動車、歩行者)  
\#3) 224×448(自動車、歩行者)  
\#4) 960×1952(歩行者のみ)  
\#5) 544×1120(自動車、歩行者)  

※<u>GTX1060のGPUメモリーが不足しているため、一部のモデルは解像度を下げて学習を行いました。</u>

以上の各モデルを30~50エポック実行します（過学習が発生し始めたらそこで止めてもかまいません）  
各モデルの学習を２フェイズに渡って行います。  
（学習にはKerasのReduceLROnPlateauを用いて、５エポックval_lossが下がらなくなったら、  
学習率を1/2にして更に精度を上げるので、長いエポック回せば、1phaseのみでも可能かもしれません）

ここではnotebook形式ではなくpyファイルを使用しますので、コマンドラインで  
```
python ～.py  
```
と指定して実行してください。


### 1.2.1 PolyYOLOモデル＃1および＃2トレーニング
#### ＜モデル \#1 トレーニング＞  
##### (phase 1)  
yolo_v4_wo_poly_multiscale.pyを使用しますが、モデル#1とモデル#2で共用することになり、さらにphase1/phase2もあるので、  
yolo_v4_model1_phase1.py / yolo_v4_model1_phase2.pyといった名前でコピーして使用しました。

各ファイルの779行目前後にパスの指定がありますので、該当するパスを指定します。

phase：該当するphase番号を指定します(1 or 2)  
annotation_path：data_for_yolo_training.txt が格納されているパスを指定します。  
validation_path：data_for_yolo_validating.txt が格納されているパスを指定します。  
log_dir：学習済みモデルを保存するパスを指定します。（１エポック毎に別名で保存されます）  
classes_path：yolo_classes.txtが格納されているパスを指定します。  
　　　　　　　（乗用車、歩行者以外も学習させる場合は、このファイルに追記する必要があります）  
anchors_path：yolo_anchors.txtが格納されているパスを指定します。  
input_shape：モデル\#1は(448, 864) を指定します。


```python
phase = 1

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model1\phase1\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
input_shape = (448, 864) # multiple of 32, hw chleba
##input_shape = (352, 704)  # multiple of 32, hw chleba 2
```

コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model1_phase1.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel1_pahse1.h5といった名前でコピーしておきます。


##### (phase 2)  
パスの指定はほぼ同様ですが、学習済みモデルの出力先のみphase1とは分けました。

```python
phase = 2

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model1\phase2\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
input_shape = (448, 864) # multiple of 32, hw chleba
##input_shape = (352, 704)  # multiple of 32, hw chleba 2
```

更にphase2では、phase1で学習済みの最良のモデルを読み込ませるため、796行目前後のモデルの読込みにphase1のモデルのパスを指定してやります。

```python
if phase == 1:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
else:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=True,
    weights_path=r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model1\phase1\log/model1_phase1.h5')
```



コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model1_phase2.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel1_pahse1.h5といった名前でコピーしておきます。

☆これでモデル\#1での最良モデルを作成できました。


#### ＜モデル \#2 トレーニング＞  
##### (phase 1)  
モデル#1と同様にyolo_v4_wo_poly_multiscale.pyを使用するので、  
yolo_v4_model2_phase1.py / yolo_v4_model2_phase2.pyといった名前でコピーして使用しました。  

各ファイルの779行目前後にパスの指定がありますので、該当するパスを指定します。

phase：該当するphase番号を指定します(1 or 2)  
annotation_path：data_for_yolo_training.txt が格納されているパスを指定します。  
validation_path：data_for_yolo_validating.txt が格納されているパスを指定します。  
log_dir：学習済みモデルを保存するパスを指定します。（１エポック毎に別名で保存されます）  
classes_path：yolo_classes.txtが格納されているパスを指定します。  
　　　　　　　（乗用車、歩行者以外も学習させる場合は、このファイルに追記する必要があります）  
anchors_path：yolo_anchors.txtが格納されているパスを指定します。  
input_shape：モデル\#2は(352, 704)を指定します。

```python
phase = 1

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model2\phase1\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
##input_shape = (448, 864) # multiple of 32, hw chleba
input_shape = (352, 704)  # multiple of 32, hw chleba 2
```

コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model2_phase1.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel2_pahse1.h5といった名前でコピーしておきます。

##### (phase 2)  
パスの指定はほぼ同様ですが、学習済みモデルの出力先のみphase1とは分けました。

```python
phase = 2

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model1\phase2\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
input_shape = (448, 864) # multiple of 32, hw chleba
##input_shape = (352, 704)  # multiple of 32, hw chleba 2
```

更にphase2では、phase1で学習済みの最良のモデルを読み込ませるため、796行目前後のモデルの読込みにphase1のモデルのパスを指定してやります。

```python
if phase == 1:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
else:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=True,
    weights_path=r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model2\phase1\log/model2_phase1.h5')
```



コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model2_phase2.py
```
と指定して実行してください。　　
学習が終了した後、最良のモデルをmodel1_pahse2.h5といった名前でコピーしておきます。

☆これで、モデル\#2での最良モデルが作成できました。





### 1.2.2 PolyYOLOモデル＃3および5＃トレーニング
#### ＜モデル \#3 トレーニング＞  
##### (phase 1)  
<u>poly_v4_wo_poly_multiscale_v4.py</u>を使用します。(モデル#1/モデル#2のソースとは異なるので注意して下さい)  
モデル#3とモデル#5で共用することになり、さらにphase1/phase2もあるので、  
yolo_v4_model3_phase1.py / yolo_v4_model3_phase2.pyといった名前でコピーして使用しました。

各ファイルの1060行目前後にパスの指定がありますので、該当するパスを指定します。

phase：該当するphase番号を指定します(1 or 2)  
annotation_path：data_for_yolo_training.txt が格納されているパスを指定します。  
validation_path：data_for_yolo_validating.txt が格納されているパスを指定します。  
log_dir：学習済みモデルを保存するパスを指定します。（１エポック毎に別名で保存されます）  
classes_path：yolo_classes.txtが格納されているパスを指定します。  
　　　　　　　（乗用車、歩行者以外も学習させる場合は、このファイルに追記する必要があります）  
anchors_path：yolo_anchors.txtが格納されているパスを指定します。  
input_shape：モデル\#3は(224, 448) を指定します。


```python
phase = 1

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model3\phase1\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
#input_shape = (544, 1120) #chleba 5
input_shape = (224, 448)  # chleba 3
```

コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model3_phase1.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel3_pahse1.h5といった名前でコピーしておきます。


##### (phase 2)  
パスの指定はほぼ同様ですが、学習済みモデルの出力先のみphase1とは分けました。

```python
phase = 2

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model3\phase2\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
#input_shape = (544, 1120) #chleba 5
input_shape = (224, 448)  # chleba 3
```

更にphase2では、phase1で学習済みの最良のモデルを読み込ませるため、1077行目前後のモデルの読込みにphase1のモデルのパスを指定してやります。

```python
if phase == 1:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
else:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=True,
    weights_path=r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model3\phase1\log/model3_phase1.h5')
```

コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model3_phase2.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel3_pahse2.h5といった名前でコピーしておきます。

☆これでモデル\#3での最良モデルを作成できました。


#### ＜モデル \#5 トレーニング＞  
##### (phase 1)  
<u>poly_v4_wo_poly_multiscale_v4.py</u>を使用します。(モデル#1/モデル#2のソースとは異なるので注意して下さい)  
モデル#3とモデル#5で共用することになり、さらにphase1/phase2もあるので、  
yolo_v4_model5_phase1.py / yolo_v4_model5_phase2.pyといった名前でコピーして使用しました。

各ファイルの1060行目前後にパスの指定がありますので、該当するパスを指定します。

phase：該当するphase番号を指定します(1 or 2)  
annotation_path：data_for_yolo_training.txt が格納されているパスを指定します。  
validation_path：data_for_yolo_validating.txt が格納されているパスを指定します。  
log_dir：学習済みモデルを保存するパスを指定します。（１エポック毎に別名で保存されます）  
classes_path：yolo_classes.txtが格納されているパスを指定します。  
　　　　　　　（乗用車、歩行者以外も学習させる場合は、このファイルに追記する必要があります）  
anchors_path：yolo_anchors.txtが格納されているパスを指定します。  
input_shape：モデル\#5は(544, 1120)とすべきでですが、GTX1060ではメモリ不足の為、エラーになるので (256, 544) を指定しました。

```python
phase = 1

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model5\phase1\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
#input_shape = (544, 1120) #chleba 5
input_shape = (256, 544) #chleba 5
#input_shape = (224, 448)  # chleba 3
```

コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model5_phase1.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel5_pahse1.h5といった名前でコピーしておきます。

##### (phase 2)  
パスの指定はほぼ同様ですが、学習済みモデルの出力先のみphase1とは分けました。

```python
phase = 2

annotation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
validation_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model5\phase2\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
#input_shape = (544, 1120) #chleba 5
input_shape = (256, 544) #chleba 5
#input_shape = (224, 448)  # chleba 3
```

更にphase2では、phase1で学習済みの最良のモデルを読み込ませるため、1077行目前後のモデルの読込みにphase1のモデルのパスを指定してやります。

```python
if phase == 1:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
else:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=True,
    weights_path=r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model5\phase1\log/model5_phase1.h5')
```



コマンドラインで該当のパスに移動した後  
```
python yolo_v4_model2_phase2.py
```
と指定して実行してください。　　
学習が終了した後、最良のモデルをmodel1_pahse2.h5といった名前でコピーしておきます。

☆これで、モデル\#5での最良モデルが作成できました。


#### ＜モデル \#4 トレーニング＞  
モデル/#4では歩行者のみのラベルで学習を行います。

##### (phase 1)  
yolo_v4_full_res.pyを使用します。
phase1/phase2もあるので、  
yolo_v4_full_phase1.py / yolo_v4_full_phase2.pyといった名前でコピーして使用しました。


各ファイルの1067行目前後にパスの指定がありますので、該当するパスを指定します。

phase：該当するphase番号を指定します(1 or 2)  
annotation_path：data_for_yolo_training_pedestrian.txt が格納されているパスを指定します。  
validation_path：data_for_yolo_validation_pedestrian.txt が格納されているパスを指定します。  
log_dir：学習済みモデルを保存するパスを指定します。（１エポック毎に別名で保存されます）  
classes_path：yolo_classes_pedest_only.txtが格納されているパスを指定します。  
　　　　　　　（モデル/#1、/#2、/#3、/#5とは指定するファイルが異なるので注意して下さい）  
anchors_path：yolo_anchors_full_res.txtが格納されているパスを指定します。  
　　　　　　　（モデル/#1、/#2、/#3、/#5とは指定するファイルが異なるので注意して下さい）   
input_shape：モデル\#4は(960, 1952)とすべきでですが、GTX1060ではメモリ不足の為、エラーになるので (480, 960) を指定しました。

```python
phase = 1

annotation_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training_pedestrians.txt'
validation_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating_pedestrians.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model4\phase1\log/'
#classes_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes_pedest_only.txt'
#anchors_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors_full_res.txt'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes_pedest_only.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors_full_res.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
#input_shape = (960, 1952) #yeah, we go full res!
input_shape = (480, 960) #yeah, we go full res!
```

コマンドラインで該当のパスに移動した後  
```
python yolo_v4_full_phase1.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel4_pahse1.h5といった名前でコピーしておきます。

##### (phase 2)  
パスの指定はほぼ同様ですが、学習済みモデルの出力先のみphase1とは分けました。

```python
phase = 2

annotation_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training_pedestrians.txt'
validation_path = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating_pedestrians.txt'
log_dir = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model4\phase2\log/'
classes_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_classes_pedest_only.txt'
anchors_path = 'D:\SIGNATE\Signate_3rd_AI_edge_competition\yolo_anchors_full_res.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
#input_shape = (960, 1952) #yeah, we go full res!
input_shape = (480, 960)  # yeah, we go full res!
```

更にphase2では、phase1で学習済みの最良のモデルを読み込ませるため、1084行目前後のモデルの読込みにphase1のモデルのパスを指定してやります。

```python
if phase == 1:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=False)
else:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=True,
    weights_path=r'D:\SIGNATE\Signate_3rd_AI_edge_competition\training\model5\phase1\log/model4_phase1.h5')
```


コマンドラインで該当のパスに移動した後  
```
python yolo_v4_full_phase2.py
```
と指定して実行してください。  
学習が終了した後、最良のモデルをmodel4_pahse2.h5といった名前でコピーしておきます。

☆これで、モデル\#4での最良モデルが作成できました。


以上でPolyYOLOの学習は終了です。
各モデルのphase2で作成した最良モデルをまとめて
D:\SIGNATE\Signate_3rd_AI_edge_competition\models/
のフォルダにmodel1.h5～model5.h5といった名前にして保存しておきます。


# 2 リファイナートレーニング
バウンディングボックスの高精度化の為にリファイナーについて説明します。  
リファイナーを学習するため、各場運絵品ぐボックス内の画像を300x300ピクセルの平方画素に埋め込み学習をします。



## 2.1データの準備　
## 2.1.1 トレーニングデータの作成
トレーニングデータ及び検証用データの準備 　
事前に以下のリファイナー画像を保存するフォルダを作成しておく必要があります。
学習用：～\images\refiner_images\train/
検証用：～\images\refiner_images\val/


＜コード概要＞
prepare_data_refiner.ipynbを使用します。  
３番目のコードセルで入出力のパスを指定します。

out_dir：リファイナー学習用の画像を保存するパスを指定します。このフォルダは事前に作成しておく必要があります。  
path_labels：data_for_yolo_training.txtを格納しているパスを指定します。  
path_labels_out：リファイナー学習ラベル(data_for_refiner_training.txt)を保存するパスを指定します。
runs：実行回数を指定します。デフォルトで１５回です。（かなりの時間が掛かるので５回で強制停止させました。）

```python
#FOR TRAIN IMAGES
out_dir          = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images\refiner_images\train/' #the dir must exist
path_labels      = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
path_labels_out  = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_refiner_training.txt'
runs = 15
```

設定後、３番目のセルを実行します。
実行後out_dirで指定したフォルダにリファイナー学習用画像が、  
path_labels_outで指定したパスにdata_for_refiner_training.txtが作成されます。


全部でruns指定回数 × data_for_yolo_training.txtの行数 × 各行のバウンディングボックスの数  
の学習用画像とラベルファイルが出力されます。

<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/rifiner_pedestrian.jpg" width="240" height="240">
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/rifiner_car.jpg" width="240" height="240">


## 2.1.2 検証データの作成
同様にprepare_data_refiner.ipynbを使用します。  
３番目のコードセルで検証用データの入出力のパスを指定します。

out_dir：リファイナー検証用の画像を保存するパスを指定します。このフォルダは事前に作成しておく必要があります。  
path_labels：data_for_yolo_validating.txtを格納しているパスを指定します。  
path_labels_out：リファイナー検証ラベル(data_for_refiner_validation.txt)を保存するパスを指定します。
runs：実行回数を指定します。デフォルトで５回です。

```python
#FOR VALID IMAGES
out_dir          = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images\refiner_images\val/' #the dir must exist
path_labels      = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_validating.txt'
path_labels_out  = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_refiner_validation.txt'
runs = 5
```
設定後、３番目のセルを実行します。
実行後out_dirで指定したフォルダにリファイナー学習用画像が、  
path_labels_outで指定したパスに\data_for_refiner_validation.txtが作成されます。



## 2.2 リファイナーのトレーニング
2.1で作成したデータを元にリファイナーモデルを学習します。
本処理も２ステップ学習を行います。

＜コード概要＞
train_refiner.ipynbを使用します。  
２番目のセルでバッチサイズを指定します。今回はＧＰＵの都合上バッチサイズ５で実行しています。

### （step1）
３番目のコードセルでモデルの読込をコメントアウトします。

```python
#model.load_weights(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\refiner_model/refiner_ep003-loss_2.959-val_loss_2.769.h5')
```

５番目のコードセルでリファイナー用のラベルファイル指定します。
学習用ラベルと検証用ラベル両方読み込みます。

```python
with open(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_refiner_training.txt') as f:
    lines = f.readlines()

with open(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_refiner_validation.txt') as f:
    lines_val = f.readlines()
```

６番目のセルで学習済みモデルの出力先を指定します。
```python
checkpoint      = ModelCheckpoint(
r'D:\SIGNATE\Signate_3rd_AI_edge_competition\refiner_model/refiner_ep{epoch:03d}-loss_{loss:.3f}-val_loss_{val_loss:.3f}.h5',
monitor='val_loss',
save_weights_only=True,
save_best_only=False,verbose=1
)
```

以上の設定を行った後、全セルを実行します。  
過学習し始めるまで学習を行います。


### (step2)  
３番目のコードセルのコメントアウトを戻します。
step1で学習した最良のモデルのパスを指定してください。

```python
model.load_weights(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\refiner_model/refiner_ep003-loss_2.959-val_loss_2.769.h5')
```

step1同様に、過学習し始めるまで学習を行います。

以上の学習で得られた最良のモデルをrefiner_model.h5といった名前でコピーしておきます。

# 3 Classifier
続いて分類器を学習します。
ここでは、乗用車と歩行者を分類できるモデルを作成します。
分類器ではリファイナー同様バウンディングボックスの内容を300×300の平方画素に埋め込んだ画像を作成し、学習を行います。
事前に分類器の学習用の画像を格納するフォルダを作成しておく必要があります。  
まず～/images/classifier/trainフォルダを作成して下さい。  
そのtrainフォルダの中に0,1,2の３つのサブフォルダを作成します。  
この0フォルダの中に乗用車の画像が、1のフォルダに歩行者、2のフォルダにその他背景の画像が格納されます。
検証用の画像に関しても同様に、～/images/classifier/valフォルダを作成し、そのvalフォルダの中に0,1,2のサブフォルダを作成します。

## 3.1 データの準備
＜コード概要＞
prepare_data_classifier.ipynbを使用します。  
３番目のコードセルで各入出力のパスを指定します

out_dir：学習用の画像保存パスを指定  
out_dir_val：検証用の画像保存パスを指定  
path_labels：data_for_yolo_training.txtを格納しているパスを指定
generate_random：背景画像作成回数

```python
out_dir       = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images\classifier\train' #the directory has to be created!
out_dir_val   = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images\classifier\val' #the directory has to be created!
path_labels   = r'D:\SIGNATE\Signate_3rd_AI_edge_competition/data_for_yolo_training.txt'
generate_random = 12 #negative images
```

全セルを実行すると、指定したフォルダに学習用画像及び検証用画像が出力されます

(乗用車)  
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/classifier_class0.jpg"  width="240" height="240">  
(歩行者)  
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/classifier_class1.jpg"  width="240" height="240">  
(その他背景)  
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/classifier_class2.jpg"  width="240" height="240">  


## 3.2 分類器トレーニング
＜コード概要＞  

train_classifier.ipynbを使用します。  
２番目のコードセルで各入出力のパスを指定します

batch_size_all：バッチサイズを指定します。
epochs_all：エポック数を指定します。
imgs_train：3.1で作成した学習画像の層数を指定します。
imgs_val：3.1で作成した学習画像の層数を指定します。
path_train：学習画像のパスを指定します。
path_val：検証画像のパスを指定します。


```python
batch_size_all    = 5
epochs_all        = 100
imgs_train        = 762253
imgs_val          = 40497
path_train        = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images\classifier\train/'
path_val          = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images\classifier\val/'
```
４番目のセルの学習済みモデルの読込みをコメントアウトします。

```python
#model.load_weights(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\classifier_model\effnet_ep005-loss67.592-val_loss46.489.h5')
```

５番目のセルのModelCheckpointに学習済みモデルの出力先を指定します。  
```python
checkpoint      = ModelCheckpoint(
r'D:\SIGNATE\Signate_3rd_AI_edge_competition\classifier_model\effnet_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
 monitor='val_loss',
 save_weights_only=True,
 save_best_only=False,verbose=1
 )
```

１～５番目のセルを実行します。


## 3.3分類器の評価
同じくtrain_classifier.ipynbを使用します。  
最後のセルに以下の設定を行います。

target_class：０，１，２で随時実行する  
bad_imgs：誤分類したカウント数（0スタート）  
dir_imgs_name：各クラスが格納されているフォルダを指定する。  
move_path：誤検知した画像を移動するパスを指定  


```python
target_class = 2 #realize for 0, 1, 2
bad_imgs = 0
dir_imgs_name   = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\classifier_images\train/'+str(target_class)
move_path       = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\classifier_images\exclusion/'
```

target_classを０，１，２と順次変化させ、最後のセルを実行すると、誤分類した画像がmove_pathのほうに移動されます。


４番目のセルの学習済みモデルの読込みを復帰させます。

```python
model.load_weights(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\classifier_model\effnet_ep005-loss67.592-val_loss46.489.h5')
```


１～５番目のセルを再度実行し、最良の分類モデルを作成します。



# 4 Matcher
## 4.1データの準備
Machaerによって前後のフレーム間で物体を検知し、同一の物体かどうかを判断します。  
Matcharの学習には、唯一Signate提供の元データ(JSONファイルを使用します。)

＜コード概要＞  
prepare_data_matcher.ipynbを使用します。  
２番目のコードセルで各入出力のパスを指定します。  

path_labels：アノテーション用JSONファイルが格納されているパスを指定します。  
path_images：動画から切り出した静止画が格納されているパスを指定します。  
out_dir：matchaer学習用画像の出力先を指定します。  
path_out_file：matcherラベルdata_for_matcher_training.txtを出力するパスを指定します。


```python
#define paths

#here are the original train anotations in json
path_labels    = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\train_annotations'   

#path where, the are the decoded train images. here is necessary to have images in their full resolution,
#without top/bottom crop
path_images    = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images'

#path where, the are the decoded train images
out_dir        = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\images\matcher'

path_out_file  = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_matcher_training.txt'
```

３番目のコードセルで学習を行うクラスを指定します。ここでは、car(乗用車)、Pedestrian(歩行者)の２つのみとしますが、必要に応じて選択して下さい。

```python
#array that defines which classes have to be extracted.
#classes = ['Car', 'Pedestrian', 'Truck', 'Signal', 'Signs', 'Bicycle', 'Motorbike', 'Bus', 'Svehicle', 'Train']
classes = ['Car', 'Pedestrian']
```

全てのセルを実行します。

matherよう画像と、data_for_matcher_training.txtが出力されているのを確認します。  

data_for_yolo_training.txtから検証用データを切り出したのと同様に、
data_for_matcher_training.txtからtrain_00に該当する行を切り取り、data_for_matcher_validation.txtに貼り付け、検証データを作成します。

尚、作成画像のファイル名はクラス番号␣動画番号␣オブジェクトID␣ファイルカウンター(サイクリック)_.jpg  
といったファイル名になっているので、動画番号で区別を行う。



## 4.2 Matcherトレーニング
＜コード概要＞  
train_matcher.ipynbを使用します。  
４番目のコードセルでラベルファイル(学習用/検証用)のパスを指定します。


```python
with open(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_matcher_training.txt') as f:
    lines = f.readlines()

with open(r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_matcher_validation.txt') as f:
    lines_val = f.readlines()
```

最後のセルで、学習済みモデルの出力先を指定します。
```python
checkpoint = ModelCheckpoint(
r'D:\SIGNATE\Signate_3rd_AI_edge_competition\matcher_model/matcher_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
monitor='val_loss',
save_weights_only=True,
save_best_only=False,verbose=1
)
```

全セルを実行し、学習を行います。


以上で全てのモデルの学習が完了しました。  
これまで、作成して来た各モデルをD:\SIGNATE\Signate_3rd_AI_edge_competition\modelsといったフォルダに格納しておきます。  
このフォルダには以下のモデル群が格納されているものとします。  
・model1.h5  
・model2.h5  
・model3.h5  
・model4.h5  
・model5.h5  
・rifiner.h5  
・classifier.h5  
・matcher.h5  

## 2⃣ 評価
1⃣で学習した各モデルを使って評価を行いテストデータに対して、アノテーションデータを予想する。
今回はtrain_00.mp4をテストデータとして用いる。

ここからのコードは、  
[https://github.com/takatoshi-ii/diveintocode-ml/tree/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/signate_3rd_ai_edge_competition/scoring_service/src](https://github.com/takatoshi-ii/diveintocode-ml/tree/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/signate_3rd_ai_edge_competition/scoring_service/src)  
に格納している。

## 1.1 predictor.py
＜コード概要＞  
predictor.pyの中のScoringServiceクラスでのget_modelメソッドで、学習済みのモデルを登録します。  
モデルへのパスは、外部から引数で渡されるものと想定しています。
```python
    def get_model(cls, model_path='../model'):
```


以下のコードで学習済みモデルを順次登録していきます。  
まずは、PolyYOLOモデルから登録していきます。  
model_path：読込むモデルを指定します。  
model_image_size：該当するモデルの解像度を指定します。(解像度を変更した場合は、ここも変更する必要があります。)  

anchors_path：  
    - model4の場合：'yolos/yolo_anchors_full_res.txt'  
    - それ以外　　：'yolos/yolo_anchors.txt'  
classes_path：  
    - model4の場合：'yolos/yolo_classes_pedest_only.txt'  
    - それ以外　　：'yolos/yolo_classes.txt'  


```python
      cls.model.append(
          yolo.YOLO(
              model_path=model_path+'/model1.h5',
              model_image_size=(448, 864),
              score=0.5,
              iou=0.5,
              anchors_path='yolos/yolo_anchors.txt',
              classes_path='yolos/yolo_classes.txt'
          )
      )
```

続いて、リファイナー～matcherの各モデルを登録していきます。

```python
cls.model.append(load_model(model_path + '/classifier.h5'))
cls.model[-1]._make_predict_function()
cls.model.append(load_model(model_path + '/refiner.h5'))
cls.model[-1]._make_predict_function()
cls.model.append(load_model(model_path + '/matcher.h5'))
cls.model[-1]._make_predict_function()
cls.model.append(load_model(model_path + '/refiner_pedest.h5'))
cls.model[-1]._make_predict_function()
```

ん・・・？　refiner_pedestなんて学習していないんだけど・・・

恐らく歩行者に特化してリファイナーを学習したものだと考えられます。

また、_make_predict_function()に関しては、以下のQiitaを参考に  
[Kerasにおける"_make_predict_function()"の重要性](https://qiita.com/surumetic/items/0abd718b366ce2b2a0e0)

どうやらGPU上でモデルの事前コンパイルを通しておくようです。


predictメソッドは変更なし。


## 1.2 predictor.py
1.1で設定したのpredictor.pyを呼び出すモジュールrun.pyを作成します。

```python {.line-number}
from predictor import ScoringService
from os import listdir
import json

print("get_model")
ScoringService.get_model(r"D:\SIGNATE\Signate_3rd_AI_edge_competition\model")

input_folder = r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\video"
output_folder = r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\annotation"
video_file_names = [f for f in listdir(r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\video") if f.endswith(".mp4")]

print(video_file_names)

for video_file_name in video_file_names:
    full_video_file_name = "/".join([input_folder, video_file_name])
    full_json_file_name = "/".join([output_folder, video_file_name + ".json"])
    print("{} into {}".format(full_video_file_name, full_json_file_name))
    #print("\t ★★★★★predicting")

    print("predict")
    res = ScoringService.predict(full_video_file_name)
    print("\t ★★★★★saving")
    with open(full_json_file_name, "w") as f:
        json.dump(res, f)
```




## 3⃣ 動画作成
