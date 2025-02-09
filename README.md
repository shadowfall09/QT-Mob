# QT-Mob

Code for `Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization`.

## Environment Setup

Please follow the instructions in LLaMA-Factory to configure the basic environment. 

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```


Additionally, since this project requires quantization and Liger Kernel, make sure to install them separately.

```bash
pip install bitsandbytes
pip install liger-kernel
```

## Dataset

A preprocessed dataset (`foursquare_NYC`) is provided as a zip file in the `data` directory. Please unzip the file within the `data` directory to proceed.

## Training & Testing

**Note**: Due to variations in training, the results may have slight differences.

Before running the training script, you need to modify several commonly used parameters in `sft_pipeline.py`:

1. **Set GPU Devices**  
   Update the second line to specify the GPUs to use:  
   ```python
   CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
    ```
2.	**Specify Model Save Directory**  
    Modify line 15 to set the directory for saving the trained model:
    ```python
    PATH_TO_SFT_SAVE_DIR = "PATH_TO_SFT_SAVE_DIR"
    ```

3.	**Set Local Model Path**  
    Replace the local model download path in line 64 under the def `choose_model(base_model)` function.
4.	**Configure Experiment Settings**  
    After line 94, you can configure your experiment parameters.

The default parameters are designed to reproduce the main experiment results on the NYC dataset.
For more detailed parameter settings, refer to utils.py.

To train QT-Mob, simply run the following command:
```bash
python sft_pipeline.py
```
