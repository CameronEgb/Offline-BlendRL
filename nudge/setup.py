from setuptools import setup, find_packages

setup(
    name='nudge',
    version='0.5.0',
    author='Hikaru Shindo',
    author_email='hikisan.gouv',
    packages=find_packages(),
    include_package_data=True,
    # package_dir={'': 'nudge'},
    url='tba',
    description='Neurally gUided Differentiable loGic policiEs (NUDGE)',
    long_description=long_description,
    install_requires=[
        "tyro",
        "pygame",
        "opencv-python",
        "numpy",
        "gymnasium",
        "torch"
    ],
)
