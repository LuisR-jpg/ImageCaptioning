[Paper base](https://paperswithcode.com/paper/bangla-image-caption-generation-through-cnn).

[Code](https://drive.google.com/file/d/1KgeKJ2hjKWXyn-2jf5WV50NpfBGhbLAW/view?usp=drive_link)

[Presentation](https://drive.google.com/file/d/15nQlZZ99uesOcQQz-mkYbCKra7fqVxBh/view?usp=sharing)

Predict
```
cd Documents\Git\ImageCaptioning\
conda activate pyTorch-gpu
python predict.py --checkpoint 1685838940-3-resnet50-checkpoint.pth --path "C:\Users\lalor\Downloads\Deep Learning\ImageCLEF\ImageCLEFmedical_Caption_2023_valid_images\valid\ImageCLEFmedical_Caption_2023_valid_000001.jpg"
```