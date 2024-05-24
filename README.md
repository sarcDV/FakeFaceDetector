# FakeFaceDetector

This is a PyTorch project for training a ResNet-18 (or any other favourite) model on a dataset using binary cross-entropy loss. 
The dataset contains real and fake face images.
The script defines the model, loads the dataset, creates data loaders, configures the training
session, and trains the model. Here's a breakdown of the main components:

1. **Importing necessary libraries**: The script imports PyTorch, TorchVision, TorchUtil, and other libraries for image processing, transformations, and logging.
2. **Setting up the environment**:
        * The `device` variable is set to use the GPU (if available) or CPU for computations.
        * The `log_path`, `trainID`, and `save_path` variables are set to store logs and models.
3. **Loading datasets**: The script defines a function `load_datasets` to load the dataset from a root directory, applying transformations as needed.
4. **Creating data loaders**: The script creates data loaders for training, validation, and testing using the `DataLoader` class from TorchUtil.
5. **Defining the model**: The script defines a ResNet-18 model using TorchVision's `models.resnet18()` function.
6. **Configuring the training session**:
        * The `start_epoch` variable is set to keep track of the current epoch number.
        * The `best_loss` variable is set to keep track of the best loss value during training.
7. **Training loop**: The script iterates through epochs, performing the following steps:
        + Train the model using binary cross-entropy loss with the `loss_func()` function.
        + Log the training loss and accuracy using TensorBoard.
        + Save the model's state dictionary at regular intervals using the `torch.save()` function.

Some notable features of this script include:

* **TensorBoard logging**: The script uses TensorBoard to log training metrics, such as loss and accuracy, during each epoch.
* **Model checkpointing**: The script saves the model's state dictionary at regular intervals, allowing for easy recovery in case of training interruptions or crashes.
* **Validation loop**: The script also includes a validation loop to track performance on the validation set during training.
