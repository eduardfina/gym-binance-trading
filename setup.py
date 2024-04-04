from setuptools import setup, find_packages

setup(
    name='crypto-investment-agent-TFM',
    version='0.0.1',
    packages=find_packages(),

    author='Eduard López',
    author_email='leduard787@gmail.com',
    license='MIT',

    install_requires=[
        'gymnasium>=0.29.1',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.1.1'
    ],

    package_data={
        'gym_anytrading': ['datasets/data/*']
    }
)