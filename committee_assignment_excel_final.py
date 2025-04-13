############################################################
# committee_assignment_excel_final.py
############################################################

import pulp
import pandas as pd

def solve_committee_assignment(
    n, 
    m, 
    R,
    min_committee_sizes=None,
    max_committee_sizes=None,
    global_min=None,
    global_max=None
):
    """
    Solve the committee assignment problem via Integer Linear Programming.

    You can specify EITHER per-committee min/max arrays OR a global min/max 
    (or both). We'll merge them as follows:
      - final_min_j = max(min_committee_sizes[j], global_min) if global_min is not None
      - final_max_j = min(max_committee_sizes[j], global_max) if global_max is not None

    :param n:  int, number of students
    :param m:  int, number of committees
    :param R:  2D list of shape (n x m),
               R[i][j] = rating of assigning student i to committee j

    :param min_committee_sizes: list of length m, or None
    :param max_committee_sizes: list of length m, or None
    :param global_min: int or None
    :param global_max: int or None

    :return: (max_rating, assignment),
             max_rating = float, sum of ratings
             assignment = list of length n, assignment[i] = j means
                         student i is assigned to committee j
    """
    if min_committee_sizes is None:
        min_committee_sizes = [0]*m
    if max_committee_sizes is None:
        max_committee_sizes = [n]*m

    if len(min_committee_sizes) != m or len(max_committee_sizes) != m:
        raise ValueError("min_committee_sizes and max_committee_sizes must be length m")

    L = []
    U = []
    for j in range(m):
        if global_min is not None:
            final_min = max(min_committee_sizes[j], global_min)
        else:
            final_min = min_committee_sizes[j]
        if global_max is not None:
            final_max = min(max_committee_sizes[j], global_max)
        else:
            final_max = max_committee_sizes[j]
        L.append(final_min)
        U.append(final_max)

    if any(L[j] > U[j] for j in range(m)):
        raise ValueError("Each L[j] must be <= U[j].")
    if sum(L) > n:
        raise ValueError("Sum of all minimums (L) exceeds total students.")
    if sum(U) < n:
        raise ValueError("Sum of all maximums (U) is less than total students.")

    # Create the problem
    prob = pulp.LpProblem("CommitteeAssignment", pulp.LpMaximize)

    # Decision variables: x[i][j] in {0,1}
    x = pulp.LpVariable.dicts("x", (range(n), range(m)), cat=pulp.LpBinary)

    # Objective
    prob += pulp.lpSum(R[i][j] * x[i][j] for i in range(n) for j in range(m)), "MaximizeRatings"

    # Each student assigned exactly once
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(m)) == 1, f"Student_{i}_OneCommittee"

    # Committee size constraints
    for j in range(m):
        prob += pulp.lpSum(x[i][j] for i in range(n)) >= L[j], f"Committee_{j}_Min"
        prob += pulp.lpSum(x[i][j] for i in range(n)) <= U[j], f"Committee_{j}_Max"

    # Solve with no solver logs
    solver = pulp.PULP_CBC_CMD(msg=0)  # <--- msg=0 hides the CBC logs
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]
    if status_str != 'Optimal':
        raise RuntimeError(f"No optimal solution found! Solver status: {status_str}")

    max_rating = pulp.value(prob.objective)
    assignment = [None]*n
    for i in range(n):
        for j in range(m):
            if pulp.value(x[i][j]) > 0.5:
                assignment[i] = j
                break

    return max_rating, assignment

def solve_from_excel(
    excel_file_path,
    output_excel_path="results.xlsx",
    global_min=None,
    global_max=None,
    min_committee_sizes=None,
    max_committee_sizes=None
):
    """
    Reads 'excel_file_path' with columns: [Name, Comm1, Comm2, ..., CommM].
    - The first column is the student name/ID.
    - The remaining columns are committee ratings.

    Solves using solve_committee_assignment (with either global or 
    per-committee min/max sizes).

    Exports result to 'output_excel_path' with each committee 
    in its own column.

    :param excel_file_path:   str, path to input .xlsx
    :param output_excel_path: str, path to output .xlsx
    :param global_min:        int or None
    :param global_max:        int or None
    :param min_committee_sizes: list[int] of length M or None
    :param max_committee_sizes: list[int] of length M or None

    :return: (best_score, assignment, student_names)
    """
    df = pd.read_excel(excel_file_path)
    student_names = df.iloc[:, 0].tolist()
    rating_cols = df.iloc[:, 1:]
    n = len(df)
    m = rating_cols.shape[1]
    R = rating_cols.values.tolist()

    best_score, assignment = solve_committee_assignment(
        n=n,
        m=m,
        R=R,
        min_committee_sizes=min_committee_sizes,
        max_committee_sizes=max_committee_sizes,
        global_min=global_min,
        global_max=global_max
    )

    # Build a summary
    committees_assigned = [[] for _ in range(m)]
    committee_ratings = [0]*m
    for i, j in enumerate(assignment):
        committees_assigned[j].append(student_names[i])
        committee_ratings[j] += R[i][j]

    col_data = []
    max_col_len = 0
    committee_headers = rating_cols.columns.tolist()

    for j in range(m):
        column_list = []
        # Committee name
        column_list.append(committee_headers[j])
        # Total rating
        column_list.append(f"Total rating: {committee_ratings[j]}")
        # Number assigned
        column_list.append(f"Number assigned: {len(committees_assigned[j])}")
        # Students
        column_list.extend(committees_assigned[j])
        max_col_len = max(max_col_len, len(column_list))
        col_data.append(column_list)

    for j in range(m):
        if len(col_data[j]) < max_col_len:
            col_data[j].extend([""] * (max_col_len - len(col_data[j])))

    df_export = pd.DataFrame({
        f"{committee_headers[j]}": col_data[j] for j in range(m)
    })

    with pd.ExcelWriter(output_excel_path) as writer:
        df_export.to_excel(writer, index=False, sheet_name="Assignments")

    return best_score, assignment, student_names

if __name__ == "__main__":
    excel_input = "students_committees.xlsx"
    excel_output = "results.xlsx"

    # Example usage: global min=2, max=10
    global_min_value = 2
    global_max_value = 10

    min_sizes = None
    max_sizes = None

    best_score, assignment, names = solve_from_excel(
        excel_file_path=excel_input,
        output_excel_path=excel_output,
        global_min=global_min_value,
        global_max=global_max_value,
        min_committee_sizes=min_sizes,
        max_committee_sizes=max_sizes
    )

    print(f"Optimal Total Rating: {best_score}")
    print(f"Results exported to {excel_output}")
