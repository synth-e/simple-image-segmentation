# Simple image segmentation

Install requirements

```bash
pip install torch torchvision
```

Run with your dataset
```sh
python main.py --data <data_path> --output <model_output_path> --num_classes <num_classes> --batch_size <default=8> --num_epochs <default=10>
```

- `<data_path>` : Path to dataset folder, expects to include `images/` and `masks/` and follow VOC format
- `<num_classes>`: Number of objects, the value in masks must be strictly in range `[0, num_classes-1]`
- `<model_output_path>` : Path to save trained model


Predict image with trained model
```sh
python predict.py --model <model_path> --image <image_path> --num_classes <num_classes>
```