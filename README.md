# Technology Evaluation Data Analyzer

This project is a modular Python system for generating, slicing, and analyzing multidimensional technology evaluation data.  
It helps **visualize and compare technologies** across different dimensions (1D–4D), revealing how features like scalability, automation, and efficiency vary across techs, groups, and environments.

---

## 🧠 Overview

The system is composed of three main modules:

### 1. `data_generator.py`
Generates structured NumPy arrays representing:
- **Technology groups** (e.g., AI systems, hardware, energy solutions)
- **Feature weights and scores**
- **Multidimensional data** (0D to 4D arrays) that encode attributes, performance metrics, and evaluation results
 
### 2. `data_analyzer.py`
Provides slicing and analytical tools to: 
- Extract feature-level data from any dimension (1D–4D)
- Reverse, filter, or reshape slices
- Combine evaluation metrics for in-depth insight

### 3. `main.py`
The orchestration script that:
- Calls `generate_data()` from `data_generator.py`
- Uses `data_analyzer.py` to analyze and compare technologies
- Outputs organized summaries for each data dimension

## ⚙️ Features

- Modular design (generation, slicing, and evaluation are separate)
- Supports **0D → 4D NumPy arrays**
- Analyzes technology performance across **multiple environments**
- Includes feature weighting for interpretability, scalability, and efficiency
- Prepares data for AI benchmarking, visualization, and reporting

---

## 📦 Installation

Requires **Python 3.9+** and **NumPy**:

```bash
git clone https://github.com/kngk0/Tech_rater_Project.git
cd Tech_rater_Project
pip install -r requirements.txt
