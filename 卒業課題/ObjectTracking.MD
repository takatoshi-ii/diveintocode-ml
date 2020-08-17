# Signateの第3回AIエッジコンテストをPolyYOLOを使ってやってみた
<br>
<br>

Signateで物体追跡のコンペで２位に入賞されたIRAFM-AIチームが採用していたPolyYOLOを用いた手法を実際にやってみる。
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
1.1.1入力動画をフレーム単位で静止画に切り出し。  
　 動画当たり、２分（１２０秒）/５fpsなので６００フレームの画像が取得できる。  
 　提供されている動画数が２５個なので合計１５,０００個の静止画が取得できる。  
 　ただ、物体検知を学習するのに１５,０００個では少なすぎるので以下のデータ拡張を行う。  
 　その中でtrain_00.mp4の動画をvalidation用に使う。

1.1.2 JSONラベルをPolyYOLOラベル形式に変換する  
　　　画像パス␣x1,y1,x2,y2,classNo␣x1,y1,x2,y2,classNo…  
　　　（␣：半角スペース）

②PolyYOLOの学習するためにデータ拡張を行う。  
　ⅰ）インペイントを作成  
　　　ランダムで２つの画像を選択し、一方をベースとして、もう片方の画像のバウンディングボックスの情報を移植する。
<img src="https://github.com/takatoshi-ii/diveintocode-ml/blob/master/%E5%8D%92%E6%A5%AD%E8%AA%B2%E9%A1%8C/124_resize.jpg" width="80%" height="80%">  
本来ありえない場所に他のバウンディングボックスが移植されているのが分かると思う。　　
この処理を１００００画像に対して行う。

## トレーニング



## 評価



## 動画作成
