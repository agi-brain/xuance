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
                "environment/magent2/libmagent.dylib"  # for magent2 environment on MacOS (for Intel CPU)
            ]
    },
    version="1.2.0",
    description='XuanCe: A Comprehensive and Unified Deep Reinforcement Learning Library.',
    long_description='XuanCe is an open-source ensemble of Deep Reinforcement Learning (DRL) algorithm implementations. We call it as Xuan-Ce (玄策) in Chinese. "Xuan (玄)" means incredible and magic box, "Ce (策)" means policy. DRL algorithms are sensitive to hyper-parameters tuning, varying in performance with different tricks, and suffering from unstable training processes, therefore, sometimes DRL algorithms seems elusive and "Xuan". This project gives a thorough, high-quality and easy-to-understand implementation of DRL algorithms, and hope this implementation can give a hint on the magics of reinforcement learning. We expect it to be compatible with multiple deep learning toolboxes( PyTorch, TensorFlow, and MindSpore), and hope it can really become a zoo full of DRL algorithms.',
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
        "torch": ["torch==1.13.0"],
        "tensorflow": ["tensorflow==2.6.0"],
        "mindspore": ["mindspore==2.2.0"],
        "all": [
            "torch==1.13.0",
            "tensorflow==2.6.0",
            "mindspore==2.2.0"  # mindspore might be installed manually.
        ],
        "atari": ["atari-py==0.2.9",  # for Atari
                  "ale-py==0.7.5"],
        "box2d": ["box2d-py==2.3.5"],  # for box2d
    },
    install_requires=[
        "numpy>=1.21.6",
        "scipy==1.7.3",
        "PyYAML",  # default version is 6.0
        "gym==0.26.2",
        "gymnasium==0.28.1",
        "gym-notices==0.0.8",
        "pygame==2.1.0",
        "tqdm==4.62.3",
        "pyglet==1.5.15",
        "pettingzoo>=1.23.0",  # for MARL
        "tensorboard==2.11.2",  # logger
        "wandb==0.15.3",
        "moviepy==1.0.3",
        "imageio",  # default version is 2.9.0
        "opencv-python==4.5.4.58",
        "mpi4py",  # default version is 3.1.3
    ],
    setup_requires=['pytest-runner'],
    tests_requires=['pytest'],
    test_suite='tests',
)
