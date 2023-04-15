# Batch Clipper
Prompt-based batch clipping of images in directory for AUTOMATIC1111

clipseg技術を使って指定したワードに対応する部分を切り出し/マスクを行います
フォルダ内の画像すべてに同様の処置を行えます

# Install

初回起動時clipsegを自動的にインストールします
初回使用時clipsegモデルを自動的にダウンロードします

## 使い方
![sapmle0](https://github.com/hako-mikan/sd-webui-batch-clipper/blob/imgs/sample0.png)
sourceに画像を読み込みます。一括処理する前のテストを行います。
例えば人を切り出したい場合、wordsに人に関する言葉を入力します。サンプルの場合はgirlと入力しています。この状態でsingleを押すと処理が始まります。
処理が終わるとresultに結果が表示されます。結果が良ければそのままinputとoutputのフォルダ名を入力してbatchボタンを押すとフォルダ内のすべてのファイルに対して同様の処理が行われ保存されます。

## 設定
### mode
#### clip
wordで指定された領域を切り取ります。
カンマで区切ることで複数のwordに対応します。
#### mask
wordで指定された領域を白く塗りつぶします。
カンマで区切ることで複数のwordに対応します。
次の画像は「terrarium,bookshelf,sky,face」と入力した結果です。  
![sapmle1](https://github.com/hako-mikan/sd-webui-batch-clipper/blob/imgs/sample1.png)
![sapmle3](https://github.com/hako-mikan/sd-webui-batch-clipper/blob/imgs/sample3.png)

### option
cropにチェック入れると処理後に余白を削除します。
「girl」でcropした結果。  
![sapmle1](https://github.com/hako-mikan/sd-webui-batch-clipper/blob/imgs/sample2.png)
