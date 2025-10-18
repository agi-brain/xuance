Basic Configurations
--------------------------

The basic parameter configuration is stored in the "xuance/config/basic.yaml" file, as shown below:

.. code-block:: yaml

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorflow"

    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: "tensorboard", "wandb".
    wandb_user_name: "your_user_name"

    parallels: 10
    seed: 2910
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    test_mode: False
    test_steps: 2000

    device: "cpu"


It should be noted that the value of the `device` variable in the `basic.yaml` file varies depending on the specific deep learning framework, as outlined below:

| - PyTorch: "cpu", "cuda:0";
| - TensorFlow: "cpu"/"CPU", "gpu"/"GPU";
| - MindSpore: "CPU", "GPU", "Ascend", "Davinci".
