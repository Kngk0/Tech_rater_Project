# Technology Evaluation Data Analyzer

Python-based framework for evaluating technology data across multiple dimensions
Demonstrates Python scripting, NumPy array manipulation, data processing, and modular project design for scalable analysis.

---

## 🧑‍💻 Tech Used

- Python (syntax, loops, functions, file handling)
- NumPy (arrays, indexing, slicing, reshaping, copying, joining, splitting, sorting, filtering)
- Modular Python scripts for structured, reusable code

## 🧠 Overview / Purpose

This project provides a framework to:
- Generate multidimensional arrays representing technology metrics
- Manipulate and analyze arrays to evaluate performance, efficiency, and other key scores
- Support scalable experimentation with 1D–4D data structures
- Demonstrate Python data handling, modular design, and algorithmic thinking

It’s designed as a prototype for internal evaluation of technology metrics, suitable for automation and larger data analysis pipelines.

## ⚙️ Key Features / What It Does

- Array Creation: Generate 1D–4D arrays representing technology features
- Indexing & Slicing: Extract and manipulate specific subsets of data
- Data Type Handling: Convert between types, check, and define array types
- Copies & Views: Create copies and views for safe vs efficient data manipulation
- Shape & Reshape: Inspect array shapes, reshape arrays for analysis
- Iteration: Traverse arrays efficiently for calculations or checks
- Joining & Splitting: Combine and divide arrays for multi-metric analysis
- Sorting & Filtering: Organize and filter data for evaluation
- Searching: Locate specific values or subsets of data

This project focuses on building a Pythonic, modular framework for handling multidimensional data rather than on a GUI.

## 📁 Project Structure

```
tech_rater_project/
├── main.py               # Orchestrates the data generation and analysis
├── data_generator.py     # Generates 1D–4D arrays with technology metrics
├── data_analyzer.py      # Performs slicing, data type conversion, copies/views, reshaping, joining, splitting, filtering, and searching
├── README.md             # Project documentation
└── requirements.txt
```

## 📦 How to Run

1. Clone the repository:
   ```
   git clone https://github.com/Kngk0/Tech_rater_Project.git
   cd Tech_rater_Project
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

## Key Learnings / What I Implemented

- 	Python scripting for complex data manipulation
- 	NumPy array operations: indexing, slicing, reshaping, copying, joining, splitting, filtering, sorting
- 	Modular design: separate generator, analyzer, and main scripts for scalability and reusability

## Future Improvements

- Integrate with real-world datasets
- Add automated reports / CSV export of analysis
- Include visualization of multidimensional data (e.g., using Matplotlib)
- Extend framework to support machine learning input pipelines
