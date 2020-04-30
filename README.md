
# Requirements
- Python 3.7.3
```bash
pip install -r requirements.txt
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

# Dataset creation from annotated data
```
# Suppose we have images and annotation xml files in 'datasets/rawpics/{image}.[png|xml]', and save the clipped file to 'datasets/clipped'.
python scripts/clip_annotated_pics.py \
       --data-dir datasets/rawpics \
       --save-dir datasets/clipped 



# Train and test a model
```
python main.py checkpoints/tmp
```



## Tags
フォーマットは[チャンピオン名*星: アイテムのリスト]．アイテムがない場合，星1の場合はそれぞれ省略．
例) 
- レオナ星1アイテムなし = leona
- グレイブス星2にGA,BFがついている場合 = graves*2: GA, BF
- アイテム置き場に涙とBF = :tear, BF
- 盗賊のグローブをつけている場合はそこから出てきたアイテムも記入

labelImg/data/predefined_classes.txt を同プロジェクト内のpredefined_classes.txtで上書きするとアノテーションの際ラベルがサジェストされます．

