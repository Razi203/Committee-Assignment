# Committee Assignment with Excel Data

This repository provides a Python script to **assign students** to committees while **maximizing total ratings**. It enforces:

- Minimum and maximum committee sizes (globally or per committee).  
- One committee assignment per student.

The solution is formulated as an **Integer Linear Programming (ILP)** problem using [PuLP](https://github.com/coin-or/pulp), an open-source Python library for linear and integer optimization. The solver (CBC by default) finds an **optimal** solution in polynomial time for typical bipartite assignment constraints.

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Data Format (Excel)](#data-format-excel)  
4. [How to Use](#how-to-use)  
   - [Step 1: Prepare the Excel File](#step-1-prepare-the-excel-file)  
   - [Step 2: Configure min/max constraints](#step-2-configure-minmax-constraints)  
   - [Step 3: Run the Script](#step-3-run-the-script)  
   - [Step 4: Inspect the Output](#step-4-inspect-the-output)  
5. [Customization](#customization)  
6. [FAQ](#faq)  

---

## Requirements

- **Python 3.7+** (or later)
- **pip** (Python package manager)

## Installation

1. **Clone** (or download) this repository.  
2. **Install** the required Python libraries:

   ```bash
   pip install pulp pandas openpyxl
   ```

   - **pulp** – the optimization solver  
   - **pandas** – for reading/writing Excel files  
   - **openpyxl** – for handling `.xlsx` formats

---

## Data Format (Excel)

Your **input Excel file** should have:

- **Column A**: Student names (one row per student).  
- **Columns B..M**: Numerical ratings for each committee. Each column corresponds to a **different committee**.  
  - If there are \(m\) committees, you should have **\(m\) columns** of ratings after the first name column.

Example layout for an input file with 3 committees:

| **Name** | **Comm0** | **Comm1** | **Comm2** |
|----------|----------:|----------:|----------:|
| Alice    | 10        |  3        |  5        |
| Bob      |  2        |  9        |  1        |
| Carol    |  7        |  4        | 10        |
| ...      |  ...      |  ...      |  ...      |

---

## How to Use

### Step 1: Prepare the Excel File

1. Create a file (e.g., **`students_committees.xlsx`**) with the above format:
   - First column: Student names
   - Next columns: Ratings for each committee.

2. Ensure no extra blank rows or header rows that might break your data reading.

### Step 2: Configure min/max constraints

Open the script file (e.g. `committee_assignment_excel_final.py`) and look for these lines in the `if __name__ == "__main__":` section at the bottom:

```python
# Example: a single global min=2, max=10 for all committees
global_min_value = 2
global_max_value = 10

# Or define per-committee arrays (must match the number of columns of ratings).
min_sizes = None
max_sizes = None
```

You have two options:

1. **Global constraints** for all committees:  
   - Set `global_min_value` and `global_max_value` to integers (e.g. 2, 10).  
   - Set `min_sizes = None` and `max_sizes = None` so the script only uses global constraints.

2. **Per-committee constraints**:  
   - Let `min_sizes = [m1, m2, m3, ...]` (one entry per committee).  
   - Let `max_sizes = [M1, M2, M3, ...]`.  
   - Set `global_min_value = None` and `global_max_value = None`.  

3. **Mix** them:  
   - If you pass both global and per-committee arrays, the final min for committee j is `max(per-committee min[j], global_min_value)`, and the final max is `min(per-committee max[j], global_max_value)`.

### Step 3: Run the Script

After configuring constraints, run:

```bash
python committee_assignment_excel_final.py
```

What happens?

1. The script reads `students_committees.xlsx`.  
2. It sets up an ILP to **maximize** the sum of `(rating of student i in committee j) * assignment[i,j]`.  
3. It enforces your min/max committee sizes.  
4. It solves using PuLP’s default CBC solver.  
5. It saves the **results** to `results.xlsx` by default.

If you want a different input file or output file, change:

```python
excel_input = "students_committees.xlsx"
excel_output = "results.xlsx"
```

in the main block of the script.

### Step 4: Inspect the Output

The script creates **`results.xlsx`** (or your chosen filename) with one sheet named `"Assignments"`. Each committee is placed in its own column:

- Row 1: **Committee j**  
- Row 2: **Total rating** for that committee  
- Row 3: **Number of assigned students**  
- Rows 4+ : Each assigned student’s **name** in a separate row

For example (3 committees):

| Committee_0        | Committee_1         | Committee_2          |
|--------------------|----------------------|-----------------------|
| Committee 0        | Committee 1         | Committee 2          |
| Total rating: 15   | Total rating: 22    | Total rating: 18     |
| Number assigned: 2 | Number assigned: 3  | Number assigned: 1   |
| Alice              | Bob                 | Carol                |
| David              | Eve                 |                       |
|                    | Frank               |                       |

Empty cells appear if different committees have different numbers of assigned students.

---

## Customization

- **Change the Solver**: You can use alternative solvers (like **Gurobi**, **CPLEX**, or **GLPK**) if installed. Update the line `prob.solve()` with the appropriate solver command (e.g., `prob.solve(pulp.GUROBI_CMD())`).  
- **Additional Constraints**: Add any additional constraints to the ILP (e.g. a student can’t join certain committees). You can do so by adjusting the code in the `solve_committee_assignment` function.  
- **Output Format**: If you prefer each committee on its own sheet (instead of columns), you can adapt the “result export” code in `solve_from_excel`.

---

## FAQ

1. **Why do I get “No optimal solution found!”?**  
   - Check if your min constraints are too high or max constraints are too low, making the problem infeasible.  
   - Or if the solver fails for another reason, try installing a different solver.

2. **How fast does it run?**  
   - For ~100 students and ~10 committees, it usually finishes **under a second** with the default CBC solver on a standard laptop.

3. **Can I read a CSV instead of Excel?**  
   - Yes, replace `pd.read_excel(...)` with `pd.read_csv(...)` in the code. Then your data format changes accordingly.

---

**Enjoy assigning students with optimal results!** 

For questions or issues, please open an [issue](#) or contact the maintainers.