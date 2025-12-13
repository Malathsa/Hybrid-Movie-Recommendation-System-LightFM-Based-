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

## Requirements

- Python 3.x
- Libraries listed in requirements.txt

If running on GitHub Codespaces, Python is already available.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The project uses the MovieLens dataset.

Ensure the following files are available:
 - ratings.csv
 - movies.csv

These files must be placed inside the data/ directory as shown in the project structure.

---

## Running the Project
### Option 1: Run the Full Notebook (Recommended)
Open and run the notebook:
```bash
Full_implementation.ipynb
```


Run all cells sequentially to:
 - Train the recommendation model
 - Evaluate performance offline
 - Save the final model checkpoint
 - Measure inference response time

### Option 2: Run the Streamlit Interface

Launch the interactive web interface using:
```bash
Streamlit run app.py
```
### Option 3: Run from the Command Line

If supported by the project configuration:
```bash
python main.py
```
---

## How to Use the System
 1. Select the recommendation mode:
 - Existing User (User ID–based recommendations)
 - Cold-Start User (preference-based recommendations)
 2. For cold-start users, provide:
 - Preferred genres
 - Minimum acceptable rating
 - Popularity preference
 3. View the Top-N recommended movies displayed in a table format.

The system uses a pretrained Hybrid LightFM model and does not retrain during runtime.

---

## Model Checkpoint

During training, the Hybrid LightFM model uses epoch-based checkpointing.  
After each training epoch, the current model state is saved to a single checkpoint file:
```bash
lightfm_hybrid_checkpoint.pkl
```
### Notes

- Checkpointing is performed after every epoch using an epoch-based criterion 
- A single checkpoint file is maintained and continuously updated
- The architecture supports future extensions such as feedback integration and retraining

---

## License

This project is intended for academic and educational use only.
