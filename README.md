
# How to Run
```
python main.py checkpoints/tmp
```


# Annotation

動画のキャプチャに対して LabelImg (https://github.com/tzutalin/labelImg) を用いて行う．

```
git clone https://github.com/tzutalin/labelImg
rm labelImg/data/predefined_classes.txt
cp predefined_classes.txt labelImg/data/
cd labelImg
python labelImg.py

```


## Tags
フォーマットは[チャンピオン名*星: アイテムのリスト]．アイテムがない場合，星1の場合はそれぞれ省略．
例) 
- レオナ星1アイテムなし = leona
- グレイブス星2にGA,BFがついている場合 = graves*2: GA, BF
- アイテム置き場に涙とBF = :tear, BF

labelImg/data/predefined_classes.txt を同プロジェクト内のpredefined_classes.txtで上書きするとアノテーションの際ラベルがサジェストされます．

