from setuptools import setup, find_packages

setup(
    name="fashion_multitask_model",
    version="0.1.0",
    description="DeepFashion classification, attribute, retrieval & multitask models",
    author="Elene Zuroshvili",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "torch>=2.2.2",
        "torchvision>=0.17.2",
        "pillow",
        "numpy",
        "scikit-learn",
        "tqdm",
        "pytorch-metric-learning"
    ],
    entry_points={
        "console_scripts": [
            "train-classification = training.train_category:main",
            "train-attribute      = training.train_attribute:main",
            "train-retrieval      = training.train_retrieval:main",
            "train-multitask      = training.train_multi_classification:main",
            "train-multimodal     = training.train_multitask_final:main",
            "eval-fashion         = evaluation.eval:main",
        ]
    },
)
