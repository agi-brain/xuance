from setuptools import find_packages, setup

setup(
    name='xuanpolicy',
    packages=find_packages(include=['xuanpolicy', 'xuanpolicy.*']),
    package_data={
        "xuanpolicy":
            [
                "configs/*.yaml",
                "configs/*/*.yaml",
                "configs/*/*/*.yaml",
                "environment/magent2/libmagent.so"  # for magent2 environment
            ]
    },
    version='0.1.10',
    description='XuanPolicy: A Comprehensive and Unified Deep Reinforcement Learning Library.',
    author='Wenzhang Liu, Wenzhe Cai, Kun Jiang, etc.',
    author_email='',
    license='MIT',
    url='',
    download_url='',
    keywords=['deep reinforcement learning', 'software library', 'PyTorch', 'TensorFlow2', 'MindSpore'],
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.6',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    extras_require={
        "torch": ["torch==1.13.0"],
        "tensorflow": ["tensorflow==2.6.0"],
        "mindspore": ["mindspore==1.10.1"],
        "all": [
            "torch==1.13.0",
            "tensorflow==2.6.0",
            "mindspore==1.10.1"
        ]
    },
    install_requires=[
        "numpy==1.21.6",
        "scipy==1.7.3",
        "PyYAML==6.0",
        "gym==0.26.2",
        "gymnasium==0.28.1",
        "gym-notices==0.0.8",
        "box2d-py==2.3.5",
        "mpi4py==3.1.3",
        "tqdm==4.62.3",
        "pyglet==1.5.15",
        "opencv-python==4.5.4.58",  # for Atari
        "atari-py==0.2.9",
        "ale-py==0.7.5",
        "pettingzoo==1.23.0",  # for MARL
        "magent2==0.3.2",
        "tensorboard==2.11.2",  # logger
        "wandb==0.15.3",
        "moviepy==1.0.3",
        "imageio==2.9.0"
    ],
    setup_requires=['pytest-runner'],
    tests_requires=['pytest'],
    test_suite='tests',
)
