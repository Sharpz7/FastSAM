import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, "..")
sys.path.append(root_dir)

from fastsam import (
    FastSAM,
    FastSAMPrompt,
)

model = FastSAM("./weights/FastSAM.pt")
IMAGE_PATH = "./images/dogs.jpg"
DEVICE = "cpu"
everything_results = model(
    IMAGE_PATH,
    device=DEVICE,
    retina_masks=True,
    imgsz=640,
    conf=0.2,
    iou=0.9,
)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
ann = prompt_process.everything_prompt()

prompt_process.plot(
    annotations=ann,
    output_path="./output/dog.jpg",
    withContours=False,
    better_quality=True,
)
