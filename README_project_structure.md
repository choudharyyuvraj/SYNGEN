# One-DM Project Structure Guide

This document explains the organization and purpose of each folder and file in your One-DM handwriting synthesis project.

---

## Top-Level Folders & Files

- **README.md / README_project_structure.md**: Project documentation and structure guide.
- **environment.yml**: Conda environment setup (required Python packages).
- **LICENSE**: Project license.
- **parse_config.py**: Loads and parses YAML config files for training/testing.

---

## Data & Preprocessing

- **data/**: Contains the IAM dataset, vocab files, and related resources.
    - `IAM64-new/`, `IAM64_laplace/`: Images and laplace transforms.
    - `IAM64_train.txt`, `IAM64_test.txt`: Train/test splits.
    - `in_vocab.subset.tro.37`, `oov.common_words`, `unifont.pickle`: Vocabulary and symbol files.
- **data_loader/**: Data loading and preprocessing code.
    - `loader.py`: Loads IAM dataset, handles batching, style/content extraction.

---

## Model Code

- **models/**: Model architectures and components.
    - `diffusion.py`, `unet.py`, `fusion.py`, `recognition.py`, `resnet_dilation.py`, `transformer.py`, `loss.py`: Core model files.

---

## Training & Testing

- **trainer/**: Training logic.
    - `trainer.py`: Distributed/multi-GPU trainer.
    - `simple_trainer.py`: Single-device trainer for subset/CPU training.
- **train.py**: Original full training script.
- **train_finetune.py**: Fine-tuning script (optional).
- **test.py**: Original test script for IAM dataset.

---

## Subset & Custom Experiments

- **custom_5percent/**: Scripts for training/testing on a subset (1%, 5%, etc.).
    - `train_5percent.py`, `train_subset.py`: Subset training scripts.
    - `test_iam.py`, `test_single.py`: Subset testing scripts.
    - `README_5percent.md`: Guide for subset training.
- **custom/**: Scripts for custom image/text inference.
    - `test_custom.py`, `test_custom_image.py`, `setup_custom.py`: Run model on your own data.

---

## Configuration

- **configs/**: YAML config files for different training/testing setups.
    - `IAM64_5percent.yml`, `IAM64_scratch.yml`, etc.

---

## Utilities

- **utils/**: Utility functions and logging.
    - `logger.py`: Logging and TensorBoard setup.
    - `util.py`: Miscellaneous helpers.

---

## Pretrained Weights & Outputs

- **model_zoo/**: Pretrained weights and checkpoints.
    - `One-DM-ckpt.pt`, `RN18_class_10400.pth`, `vae_HTR138.pth`
- **output_5percent/**: Output logs, checkpoints, and generated images from subset training.
    - Organized by experiment name and timestamp.

---

## Documentation & Assets

- **assets/**: Images for documentation, results, and visualizations.

---

## How to Use

- **Standard Training:** Use `train.py` or `custom_5percent/train_5percent.py` for subset training.
- **Testing:** Use `test_iam.py`, `test_single.py`, or custom scripts for inference.
- **Custom Experiments:** Use scripts in `custom/` for your own images/texts.
- **Configuration:** Edit YAML files in `configs/` to change training/testing parameters.
- **Pretrained Models:** Place weights in `model_zoo/` and specify their path in scripts.

---

## Tips
- Keep only the scripts and configs you use to avoid confusion.
- Group custom and subset scripts in their respective folders for clarity.
- Use TensorBoard logs and output images to monitor training progress.

---

For more details on any file or folder, see the main README or ask for a line-by-line explanation!
