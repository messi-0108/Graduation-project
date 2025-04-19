# GCLMI: Graph Contrastive Learning with Minimum-Maximum Mutual Information

## Overview

GCLMI is a graph-based contrastive learning method designed for learning graph representations by optimizing the minimum-maximum mutual information (MMI) principle. This repository provides code to perform unsupervised and semi-supervised learning experiments on functional Magnetic Resonance Imaging (fMRI) data using graph neural networks (GNNs) and the MMI principle. The approach aims to balance the maximization of mutual information between original graphs and their augmented views while minimizing the mutual information among these views to enhance representation diversity.

## Features

- **Unsupervised Learning:** Learn graph representations from data with no labeled information.
- **Semi-Supervised Learning:** Incorporate a small amount of labeled data to guide learning.
- **Graph Neural Networks (GNNs):** Utilize state-of-the-art GNN architectures for graph-based learning.
- **fMRI Data Analysis:** Applies graph contrastive learning to functional Magnetic Resonance Imaging (fMRI) data for brain disease diagnosis.
- **Minimum-Maximum Mutual Information:** Leverages the MMI principle to ensure both richness and diversity of learned representations.

## Installation

To get started with GCLMI, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/GCLMI.git
cd GCLMI
pip install -r requirements.txt
