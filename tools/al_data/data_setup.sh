mkdir -p data/VOC0712
mkdir -p data/VOC0712/annotations
mkdir -p data/VOC0712/images
wget
cp /data/VOCdevkit/VOC2007/JPEGImages/*.jpg data/VOC0712/images
cp /data/VOCdevkit/VOC2012/JPEGImages/*.jpg data/VOC0712/images

mkdir -p data/active_learning/coco/annotations
mkdir -p data/active_learning/voc/annotations


