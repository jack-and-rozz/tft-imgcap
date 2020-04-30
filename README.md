
## Requirements
- Python 3.7.3
```bash
pip install -r requirements.txt
ln -sf {dataset_path} datasets 
```

## Structure of dataset directory
```
datasets
├── annotated_pics
│   └── others
│       ├── 00099997.jpg
│       └── 00099997.xml
│
├─── not_annotated_pics
│   └── bestsatojp_10.9
│       ├── 2020-04-30-23-13-01.png
│       └── 2020-04-30-23-13-01.xml
└── clipped
    ├── ahri.0.png
    ├── ahri.1.png
    ...
    ├── train.csv
    ├── dev.csv
    └── test.csv
```


## Annotation

動画のキャプチャに対して LabelImg (https://github.com/tzutalin/labelImg) を用いて行う．
width:80, height: 100くらいの画像をcreateRectBoxから切り抜いてタグ付け．タグのフォーマットは後述．

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
- アイテム置き場に涙とBF = items:tear, BF
- 盗賊のグローブをつけている場合はそこから出てきたアイテムも記入

labelImg/data/predefined_classes.txt を同プロジェクト内のpredefined_classes.txtで上書きするとアノテーションの際ラベルがサジェストされます．


## Dataset creation from annotated data
```
# Suppose we have images and annotation xml files in 'datasets/rawpics/{image}.[png,jpg|xml]', and save the clipped file to 'datasets/clipped'.
python scripts/clip_annotated_pics.py \
       --data-dir datasets/annotated_pics \
       --save-dir datasets/clipped
```


## Train and test a model
```
python main.py checkpoints/tmp # test.png will be generated.
```

