## Requirements
- Python 3.7.3

```bash
# Commands for preparation
pip install -r requirements.txt
ln -sf {dataset_path} datasets # Images and annotations are managed in Dropbox.
```

## How to use
```
# training
python train.py checkpoints/tmp --label-types=championg

# evaluation
python test.py checkpoints/tmp --output-dir=evals --label-types=championg

# test with screenshots
python clip_rawpics.py
```


## Structure of dataset directory
```
datasets
├── annotated_pics
│   └── bestsatojp_33664889 # The name of videos ({twitch_id}_{video_id}).
│       ├── 00099997.jpg # Screenshots of TFT games.
│       └── 00099997.xml # Annotation logs generated by LabelImg to the corresponding screenshot above.
│
├─── not_annotated_pics # Unannnotated screenshots are stored here for distinction.
│   └── bestsatojp_39482797
│       └── 2020-04-30-23-13-01.png 
└── clipped # Generated by 'scripts/clip_annotated_pics.py'.
    ├── ahri.0.png # Clipped pictures based on annotations.
    ├── ahri.1.png
    ...
    ├── train.csv  # Summaries of each picture's labels, and data separation.
    ├── dev.csv
    └── test.csv
```


## Annotation

動画のキャプチャに対して LabelImg (https://github.com/tzutalin/labelImg) を用いて行う．
labelImg/data/predefined_classes.txt を同プロジェクト内のpredefined_classes.txtで上書きするとアノテーションの際ラベルがサジェストされます．
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
- 
- レオナ星1アイテムなし = leona
- グレイブス星2にGA,BFがついている場合 = graves*2: GA, BF
- アイテム置き場に涙 = items:tear
  * 複数ある場合はとりあえず別々に付けることにする (2020/06/07)
- 盗賊のグローブをつけている場合はそこから出てきたアイテムも記入 = leona: IE, GA, thiefsglove



## Dataset creation from annotated data
```
# Suppose we have images and annotation xml files in 'datasets/rawpics/{image}.[png,jpg|xml]', and save the clipped file to 'datasets/clipped'.
python scripts/clip_annotated_pics.py \
       --data-dir datasets/annotated_pics \
       --save-dir datasets/clipped
```


