# QT-Mob

Code for `Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization`.

## Environment Setup

Please follow the instructions in LLaMA-Factory to configure the basic environment. 

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 25da686
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

**Note**: Due to training variability, results may slightly differ across runs.

### Step 1: Train the Vector Quantization Module

Please adjust the parameters in `data_process/index_pipeline.py` before running the script.  
**Note**: The released dataset already includes a pre-trained index, so this step is not required unless you wish to retrain the index yourself.

### Step 2: Train the QT-Mob Model
Before running the training script, you need to modify several commonly used parameters in `sft_pipeline.py`:

1. **Set GPU Devices**  
   Update line 2 to specify the GPUs to use:  
   ```python
   CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
    ```
2.	**Specify Model Save Directory**  
    Modify line 15 to set the directory for saving the trained model:
    ```python
    PATH_TO_SFT_SAVE_DIR = "PATH_TO_SFT_SAVE_DIR"
    ```

3.	**Set Local Model Path**  
    Replace the local model download path in line 65 under the def `choose_model(base_model)` function.
4.	**Configure Experiment Settings**  
    After line 95, you can configure your experiment parameters.

The default parameters are designed to reproduce the main experiment results on the NYC dataset.
For more detailed parameter settings, refer to utils.py.

To train QT-Mob, simply run the following command:
```bash
python sft_pipeline.py
```
