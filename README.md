# Rooftop Instance Segmentation using TensorFlow
### Aerial Imagery Dataset provided by National Topographic Office of New Zealand

Sample Outputs (downsized):

<img src="sample_out(0).png" alt="Result" width="200"> <img src="sample_out(1).png" alt="Result" width="200"/>
<img src="sample_out(2).png" alt="Result" width="200"/>

Link: https://www.airs-dataset.com/

Extract the dataset completely, store in this format:

```bash
# Training images
root/data/train/image
# Labels
root/data/train/label
# Test samples
root/data/test
```

Model Used: VGG-16, Instance Segmentation

The script was written on older version of TensorFlow (1.15.x and lower). 

Some newer python versions do not include the listing of older TF. 

You can downgrade to Python 3.6.x to use it, or use a python virtualenv to install specific python binary (recommended).

Install deps with the following command.

```python
pip install -r requirements.txt
```

Trained on Nvidia Quadro GP100 with 16GB VRAM. Batch Size:2 Input resolution 3584x3584.

Training Time: 12 hours


