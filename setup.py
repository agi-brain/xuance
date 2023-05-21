from setuptools import find_packages, setup

setup(
    name='xuanpolicy',
    packages=find_packages(include=['xuanpolicy', 'xuanpolicy.*']),
    package_data={"xuanpolicy": ["configs/*.yaml", "configs/*/*/*.yaml"]},
    version='0.1.4',
    description='XuanPolicy: A Comprehensive Deep Reinforcement Learning Library.',
    author='Wenzhang Liu, Wenzhe Cai, Kun Jiang, etc.',
    author_email='',
    license='MIT',
    url='',
    download_url='',
    keywords=['deep reinforcement learning', 'software library', 'platform'],
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
    install_requires=[
        "numpy >= 1.19.5",
        "scipy >= 1.7.3",
        "PyYAML >= 6.0",
        "gym >= 0.21.0",
        "gymnasium >= 0.28.1",
        "mpi4py >= 3.1.3",
        "tqdm >= 4.0",
        "pyglet >= 1.5.15",
        "torch >= 1.13.0",  # for PyTorch users
        # "opencv-python >= 4.5.4.58",  # for Atari
        # "mindspore",  # for MindSpore users
        # "tensorflow >= 2.6.0"  # for TensorFlow2.0 Users
        # "tensorboard >= 2.11.2"  # logger
    ],
    setup_requires=['pytest-runner'],
    tests_requires=['pytest'],
    test_suite='tests',
)
