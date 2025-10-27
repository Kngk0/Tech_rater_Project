# Technology Evaluation Data Analyzer

This project is a modular Python system for generating, slicing, and analyzing multidimensional technology evaluation data.  
It helps visualize and compare different technologies, features, and performance metrics across 1D–4D data structures.

---

## 🧠 Overview

The system is composed of three main modules:

### 1. `data_generator.py`
Generates labeled NumPy arrays representing:
- **Technology groups** (e.g., AI systems, hardware, energy solutions)
- **Feature weights and scores**
- **Multidimensional data** (0D to 4D arrays) that encode attributes, performance metrics, and evaluation results
 
### 2. `data_analyzer.py`
Provides analysis and slicing utilities for the generated data. 
It can:
- Extract detailed feature-level data across multiple dimensions
- Reverse or filter slices for deeper insight
- Combine evaluation scores (e.g., interpretability, scalability, efficiency)

### 3. `main.py`
The central orchestration script that:
- Calls `generate_data()` from `data_generator.py`
- Uses the slicing and analysis functions from `data_analyzer.py`
- Prints structured analysis results for each dimensional slice (1D to 4D)

## ⚙️ Features

- Modular architecture with clear separation of data generation and analysis
- Handles **1D, 2D, 3D, and 4D NumPy arrays**
- Built-in evaluation metrics such as interpretability, efficiency, and automation
- Reversible slicing and selective feature extraction
- Ready to extend for AI benchmarking, technology scoring, or visualization pipelines

---

## 📦 Installation

Make sure you have **Python 3.9+** and **NumPy** installed.

```bash
git clone https://github.com/YourUsername/Tech_rater_Project.git
cd Tech_rater_Project
pip install -r requirements.txt
