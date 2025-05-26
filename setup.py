#!/usr/bin/env python3
"""
Setup file for ATLAS Human-Like Enhancement System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version information
__version__ = "1.0.0"

setup(
    name="atlas-human-enhancements",
    version=__version__,
    author="ATLAS Development Team",
    author_email="contact@atlas-ai.org",
    description="Human-like cognitive enhancements for ATLAS-Qwen consciousness monitoring system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/atlas-human-enhancements",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/atlas-human-enhancements/issues",
        "Documentation": "https://github.com/yourusername/atlas-human-enhancements/docs",
        "Source Code": "https://github.com/yourusername/atlas-human-enhancements",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "line-profiler>=3.5.0",
            "memory-profiler>=0.60.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
            "mkdocstrings>=0.19.0",
        ],
        "viz": [
            "plotly>=5.10.0",
            "wandb>=0.13.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "multimodal": [
            "opencv-python>=4.6.0",
            "librosa>=0.9.0",
            "soundfile>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "atlas-demo=final_human_features:main",
            "atlas-test=comprehensive_test:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "consciousness",
        "cognitive science",
        "human-like AI",
        "emotional intelligence",
        "social cognition",
        "machine learning",
        "neural networks",
        "psychology",
        "neuroscience",
        "ethics",
        "moral reasoning",
        "temporal reasoning",
        "attention management",
        "personality modeling",
        "episodic memory",
        "intuition",
        "metacognition",
        "dreams",
        "subconscious processing",
    ],
)
