# Simple image segmentation

Install requirements

```bash
pip install torch torchvision
```

Run with your dataset
```sh
python main.py --data <data_path> --label <your_data_mapping> --output <model_output_path> --batch_size <default=8> --num_epochs <default=10>
```

- `<data_path>` : Path to dataset folder, expects to include `images/` and `masks/` and follow VOC format
- `<your_data_mapping>` : Data mapping for the dataset, expects json file.
- `<model_output_path>` : Path to save trained model

An example of `<your_data_mapping>` file:

`lable_mapping.json`
```json
{
    0: background
    1: car
    2: cat
    3: dog
    ...
}
```
