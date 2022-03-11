from setuptools import setup
from codecs import open
from os import path
from setuptools import find_packages

here = path.abspath(path.dirname(__file__))

# long_description(後述)に、GitHub用のREADME.mdを指定
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='traderl',
    packages=find_packages(),

    version='1.0.2',

    license='Apache-2.0 License',

    install_requires=['numpy', 'tensorflow', "ta", "pandas", "pandas_datareader"],

    author='komo135',
    author_email='komoootv@gmail.com',

    url='https://github.com/komo135/',  # パッケージに関連するサイトのURL(GitHubなど)

    description='Reinforcement learning is used to learn trades.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='traderl',

    classifiers=[
        'License :: OSI Approved :: Apache-2.0 License',
        'Programming Language :: Python :: 3.7',
    ],
)
