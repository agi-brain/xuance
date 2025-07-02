from setuptools import find_packages, setup

setup(
    name='xuance',
    packages=find_packages(include=['xuance', 'xuance.*']),
    package_data={
        "xuance":
            [
                "configs/*.yaml",
                "configs/*/*.yaml",
                "configs/*/*/*.yaml",
                "environment/magent2/libmagent.so",  # for magent2 environment on linux
                "environment/magent2/magent.dll",  # for magent2 environment on Windows
                "environment/magent2/libmagent.dylib"  # for magent2 environment on macOS (for Intel CPU)
            ]
    },
    version="1.3.1",
    description='XuanCe: A Comprehensive and Unified Deep Reinforcement Learning Library.',
    long_description='XuanCe is an open-source ensemble of Deep Reinforcement Learning (DRL) algorithm implementations. We call it as Xuan-Ce (玄策) in Chinese. "Xuan (玄)" means incredible and magic box, "Ce (策)" means policy. DRL algorithms are sensitive to hyperparameters tuning, varying in performance with different tricks, and suffering from unstable training processes, therefore, sometimes DRL algorithms seems elusive and "Xuan". This project gives a thorough, high-quality and easy-to-understand implementation of DRL algorithms, and hope this implementation can give a hint on the magics of reinforcement learning. We expect it to be compatible with multiple deep learning toolboxes( PyTorch, TensorFlow, and MindSpore), and hope it can really become a zoo full of DRL algorithms.',
    author='Wenzhang Liu, et al.',
    author_email='liu_wzh@foxmail.com',
    license='MIT',
    url='',
    download_url='https://github.com/agi-brain/xuance.git',
    keywords=['deep reinforcement learning', 'software library', 'PyTorch', 'TensorFlow2', 'MindSpore'],
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.6',  # Specify which python versions that you want to support
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    extras_require={
        "torch": ["torch",
                  "torchvision"],
        "tensorflow": ["tensorflow==2.6.0",
                       "tensorflow-addons==0.21.0",
                       "tensorflow-probability==0.14.0"],
        "mindspore": ["mindspore==2.2.0"],
        "all": [
            "torch",
            "tensorflow==2.6.0",
            "mindspore==2.2.14"  # mindspore might be installed manually.
        ],
        "tune": [
            "optuna>=4.1.0",
            "optuna-dashboard>=0.17.0",
            "plotly>=5.24.1",
        ],
        "atari": ["gymnasium[accept-rom-license]==1.1.1",
                  "gymnasium[atari]==1.1.1",
                  "ale-py==0.10.1"],
        "box2d": ["swig==4.3.0",
                  "gymnasium[box2d]==1.1.1"],  # for Box2D
        "minigrid": ["minigrid==3.0.0"],
        "metadrive": ["metadrive"],
        "rware": ["rware"],
        "einops": ["einops==0.8.1"],  # default version is 0.8.1 for ViT
    },
    install_requires=[
        "numpy",  # suggest version: >=1.21.6
        "scipy",  # suggest version: >=1.15.3
        "PyYAML",  # suggest version: 6.0
        "gymnasium",  # suggest version: >=1.1.1
        "pygame",  # suggest version: >=2.1.0
        "tqdm",  # suggest version: >=4.66.3
        "pyglet==1.5.15",  # suggest version: ==1.5.15
        "pettingzoo",  # for MARL, suggest version: >=1.23.0
        "tensorboard",  # logger, suggest version: >=2.11.2
        "wandb",  # suggest version: >=0.15.3
        "moviepy",  # suggest version: >=1.0.3
        "imageio",  # suggest version: 2.9.0
        "opencv-python",  # suggest version: 4.5.4.58
        "mpi4py",  # suggest version: 3.1.3
        "torch",
        "torchvision"
    ],
    setup_requires=['pytest-runner'],
    tests_requires=['pytest'],
    test_suite='tests',
)
