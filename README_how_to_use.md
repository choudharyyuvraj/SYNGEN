# How to Use One-DM

This guide explains how to set up, train, test, and run custom experiments with the One-DM handwriting synthesis project.

---

## 1. Setup Environment
- Install Anaconda or Miniconda (if not already installed)
- Create and activate the environment:
  ```cmd
  conda env create -f environment.yml
  conda activate torch_basic
  ```

---

## 2. Prepare Data
- Place the IAM dataset and vocab files in the `data/` folder as described in the project structure guide.

---

## 3. Train the Model
- **Quick experiments (1% or 5% of data):**
  ```cmd
  python custom_5percent/train_5percent.py --cfg configs/IAM64_5percent.yml --device cpu --subset_ratio 0.01
  ```
- **Full training:**
  ```cmd
  python train.py --cfg configs/IAM64_scratch.yml --device cuda
  ```

---

## 4. Test the Model
- **On IAM dataset:**
  ```cmd
  python custom_5percent/test_iam.py --cfg configs/IAM64_5percent.yml --device cpu
  ```
- **On a single custom image:**
  ```cmd
  python custom_5percent/test_single.py --image_path test.jpg --model_checkpoint <path_to_checkpoint.pt> --device cpu
  ```
- **On your own data/texts:**
  ```cmd
  python custom/test_custom_image.py --image_path my_image.png --model_checkpoint <path_to_checkpoint.pt> --device cpu
  ```

---

## 5. Monitor Training
- Use TensorBoard to visualize logs:
  ```cmd
  tensorboard --logdir output_5percent/IAM64_5percent/<experiment_folder>/tboard
  ```

---

## 6. Check Outputs
- Model checkpoints, logs, and generated images will be saved in the `output_5percent/` folder (or as specified in your config).

---

## 7. Tips
- Edit YAML files in `configs/` to change training/testing parameters.
- Place pretrained weights in `model_zoo/` and specify their path in scripts.
- Use scripts in `custom/` for your own images/texts.
- Clean up unused scripts and configs for clarity.

---

For more details, see the main README or ask for a line-by-line explanation!
