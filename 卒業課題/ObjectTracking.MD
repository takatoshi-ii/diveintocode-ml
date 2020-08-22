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
prepare_data_refiner.ipynbを使用します。　　

＜コード概要＞
mosaic_data.ipynbを使用します。  
３番目のコードセルで入出力のパスを指定します。

out_dir：リファイナー学習用の画像を保存するパスを指定します。このフォルダは事前に作成しておく必要があります。  
path_labels：data_for_yolo_training.txtを格納しているパスを指定します。  
path_labels_out：リファイナーラベル(data_for_refiner_training.txt)を保存するパスを指定します。
runs：実行回数を指定します。デフォルトで１５回です。（かなりの時間が掛かるので５回で強制停止させました。）

```python
#FOR TRAIN IMAGES
out_dir          = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\refiner_images\train/' #the dir must exist
path_labels      = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_yolo_training.txt'
path_labels_out  = r'D:\SIGNATE\Signate_3rd_AI_edge_competition\data_for_refiner_training.txt'
runs = 15
```
全部でruns指定回数 × data_for_yolo_training.txtの行数 × 各行のバウンディングボックスの数  
の学習用画像とラベルファイルが出力されます。

<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/rifiner_pedestrian.jpg">
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/image/rifiner_car.jpg">





# 3 Classifier




# 4 Matcher




## 評価



## 動画作成
