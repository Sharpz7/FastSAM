# for every image in archive
# --text_prompt "Pelvis Bone X-Ray Left and Right Side"

sudo rm -r ./output
mkdir ./output

for f in ./archive/*.jpg
do
  echo "Processing $f file..."
  python3 Inference.py --model_path ./weights/FastSAM.pt --img_path $f --imgsz 640 --iou 0.7 --conf 0.2
done