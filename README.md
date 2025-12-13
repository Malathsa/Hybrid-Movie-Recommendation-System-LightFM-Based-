# Hybrid Movie Recommendation System (LightFM-Based)

A hybrid movie recommendation system built using LightFM, combining collaborative filtering with content-based features to generate personalized movie recommendations for both existing users and cold-start users.

---

## Motivation

As online movie libraries continue to expand, users often face difficulty discovering content that truly matches their interests. Many recommendation systems rely heavily on historical interaction data, which limits their effectiveness for new users.  
This project aims to build a practical, lightweight, and deployable recommendation system that delivers meaningful recommendations immediately, even with minimal user information.

---

## Repository Overview

This repository contains:
- A Hybrid LightFM-based recommendation model
- A Streamlit interface for interactive recommendations
- A complete Jupyter Notebook that includes the entire project implementation

---

## Important Notebook

### Full_implementation.ipynb

This notebook contains the full project code in one file, including:
- Data loading
- Feature preparation
- Model training
- Offline evaluation
- Model checkpoint saving
- Response-time measurement during inference

It is provided to simplify running, understanding, and reusing the project without navigating multiple source files.

---

---

## Requirements

- Python 3.x
- Libraries listed in requirements.txt

If running on GitHub Codespaces, Python is already available.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
