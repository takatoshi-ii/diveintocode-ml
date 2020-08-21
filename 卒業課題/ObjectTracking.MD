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
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/flow.png">

①入力動画をフレーム単位で静止画に切り出し  
②複数のPloy-YOLO検出器で学習(PolyYOLO a ～ e まで５つの検出器の学習が必要)  
③WBF(Weighted Box Fusion)でアンサンブル学習  
④Refiner:検出境界を高精度化  
⑤クラス分類(自動車、歩行者 分類)  
⑥フレーム間オブジェクトの関連付け(追跡処理)  


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
　ⅰ）インペイントを作成  
　　　ランダムで２つの画像を選択し、一方をベースとして、もう片方の画像のバウンディングボックスの情報を移植します。
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/124_resize.jpg" width="80%" height="80%">  
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


　ⅱ）インペイントクロップを作成  
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








## トレーニング



## 評価



## 動画作成
