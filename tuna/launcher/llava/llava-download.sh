

wget http://images.cocodataset.org/zips/train2017.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# python vqa_download.py

unzip -d coco/ train2017.zip && unzip -d gqa/ images.zip && unzip -d textvqa/ train_val_images.zip
unzip -d vg/ images.zip.1
unzip -d vg/ images2.zip