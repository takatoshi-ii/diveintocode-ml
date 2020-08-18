# Signateの第3回AIエッジコンテストをPolyYOLOを使ってやってみた
<br>
<br>

Signateでの物体追跡のコンペで２位に入賞されたIRAFM-AIチームが採用していたPolyYOLOについてWikiにチュートリアルが詳細にまとめられていたので実際にやってみた。  
概要は以下のWikiに丁寧に分かりやすくまとめられているので(英語だけど)そちらも参考に。   
(参照)[https://gitlab.com/irafm-ai/signate_3rd_ai_edge_competition/-/tree/master](https://gitlab.com/irafm-ai/signate_3rd_ai_edge_competition/-/tree/master)

## 学習/評価フロー
PolyYOLOを用いたフローは以下の様になっている。  
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/flow.png">

①入力動画をフレーム単位で静止画に切り出し  
②複数のPloy-YOLO検出器で学習(PolyYOLO a ～ e まで５つの検出器の学習が必要)  
③WBF(Weighted Box Fusion)でアンサンブル学習  
④Refiner:検出境界を高精度化  
⑤クラス分類(自動車、歩行者 分類)  
⑥フレーム間オブジェクトの関連付け(追跡処理)  




## 前処理（データ拡張）
### 1.1.1入力動画をフレーム単位で静止画に切り出す。  
　 動画当たり、２分（１２０秒）/５fpsなので６００フレームの画像が取得できる。  
 　提供されている動画数が２５個なので合計１５,０００個の静止画が取得できる事になる。  
 　ただ、物体検知を学習するのに１５,０００個では少なすぎるので、以下のデータ拡張を行う。  
 　その中でtrain_00.mp4の動画をvalidation用に使う。
 <br>
 ＜コード＞  
 prepare_data_decode_movies.ipynbを使用する。  
 2つめのセルに以下の情報を設定  
 path_to_videosにtrain_xx.mp4があるフォルダを指定する
 path_to_save_imgsに切り出したフレーム画像を保存するフォルダを指定する


### 1.1.2 JSONラベルをPolyYOLOラベル形式に変換する  
　　　画像パス␣x1,y1,x2,y2,classNo␣x1,y1,x2,y2,classNo…  
　　　（␣：半角スペース）  
<br>
     この処理によってアノテーションデータdata_for_yolo_training.txtが指定したフォルダに作成される。  
     検証データdata_for_yolo_validating.txtを作成し、data_for_yolo_training.txtから
     /train_00/～.jpgに該当する行を切り取り、そこに張り付ける。

### 1.1.3歩行者のみを使用してPolyYOLOラベル形式を作成する  
車のクラスと歩行者のクラスの間に不均衡があるため、歩行者のみをピックアップしたアノテーションデータを作成します。

②PolyYOLOの学習するためにデータ拡張を行います。  
　ⅰ）インペイントを作成  
　　　ランダムで２つの画像を選択し、一方をベースとして、もう片方の画像のバウンディングボックスの情報を移植します。
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/124_resize.jpg" width="80%" height="80%">  
本来ありえない場所に他のバウンディングボックスが移植されているのが分かると思います。　　
この処理を１００００画像に対して行います。

## トレーニング



## 評価



## 動画作成



```flow
st=>start: Start
op1=>operation: Your Operation
cond=>condition: Yes or No?
e=>end

st->op1->cond
cond(yes)->e
cond(no)->op
```





```sequence
Title: Here is a title
A->B: Normal line
B-->C: Dashed line
C->>D: Open arrow
D-->>A: Dashed open arrow

Note left of A: Note to the\n left of A
Note over A: Note over A
Note over A,B: Note over both A and B

participant E
```
