from setuptools import setup, find_packages
import os

# 获取当前目录下的所有非Python文件作为数据文件
data_files = []
for root, dirs, files in os.walk('.'):
    if '__pycache__' in root:
        continue
    if '.git' in root:
        continue
    if root.startswith('./'):
        root = root[2:]
    for file in files:
        if not file.endswith('.py') and not file.endswith('.pyc'):
            data_files.append(os.path.join(root, file))

setup(
    name="ChessBomb_editor",
    version="2.0.1",
    description="一个使用ALNS算法的象棋炸弹谜题求解器",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': data_files,
    },
    install_requires=[
        'numpy>=1.20.0',
        'pygame>=2.0.0,<2.3.0',
        'alns>=2.0.0',
        'PyQt5>=5.15.0'
    ],
    entry_points={
        'console_scripts': [
            'chessbomb=main:main',
        ],
    },
    python_requires='>=3.8',
    author="",
    author_email="",
    url="https://github.com/YHlorra/ChessBomb_editor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)