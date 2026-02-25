import numpy as np

# Alternatives
ALTERNATIVES = {
    1: "O1: Develop novel white-box model",
    2: "O2: Reuse / adapt existing white-box model",
    3: "O3: Develop novel black-box model (limited data)",
    4: "O4: Develop novel black-box model (large data)",
    5: "O5: Reuse existing black-box model"
}

# Criteria
CRITERIA = {
    1: "C1: Accuracy",
    2: "C2: Interpretability",
    3: "C3: Development Cost/Time",
    4: "C4: Customization"
}

# Feasibility Indicators
FEASIBILITY_INDICATORS = [
    "F1: UAV platform is widely recognized and documented",
    "F2: Published white-box model exists for this UAV",
    "F3: Published black-box model exists for this UAV",
    "F4: Experimental infrastructure available (wind tunnel, etc.)",
    "F5: Capability to collect flight/energy datasets",
    "F6: Dataset volume is large and high quality"
]

def check_feasibility(f_vector):
    """
    f_vector: list [F1, F2, F3, F4, F5, F6] (0 or 1)
    Returns list of admissible alternative indices (1-5).
    """
    # Map input to F1..F6 (0-indexed)
    F = f_vector 
    
    admissible = []
    
    # O1 admissible if F4 = 1
    if F[3] == 1:
        admissible.append(1)
        
    # O2 admissible if F2 = 1
    if F[1] == 1:
        admissible.append(2)
        
    # O3 admissible if F5 = 1 and F6 = 0
    if F[4] == 1 and F[5] == 0:
        admissible.append(3)
        
    # O4 admissible if F5 = 1 and F6 = 1
    if F[4] == 1 and F[5] == 1:
        admissible.append(4)
        
    # O5 admissible if F3 = 1
    if F[2] == 1:
        admissible.append(5)
        
    return admissible

# Hardcoded Matrices from Paper (A1 to A4)
# A1: Accuracy
A1 = np.array([
    [1, 2, 5, 4, 3],
    [1/2, 1, 4, 3, 2],
    [1/5, 1/4, 1, 1/2, 1/2],
    [1/4, 1/3, 2, 1, 1],
    [1/3, 1/2, 2, 1, 1]
])

# A2: Interpretability (C2)
A2 = np.array([
    [1, 2, 6, 7, 5],
    [1/2, 1, 5, 6, 4],
    [1/6, 1/5, 1, 2, 1/2],
    [1/7, 1/6, 1/2, 1, 1/3],
    [1/5, 1/4, 2, 3, 1]
])

# A3: Development Cost/Time (C3)
A3 = np.array([
    [1, 1/3, 1/4, 1/5, 1/6],
    [3, 1, 1/2, 1/3, 1/4],
    [4, 2, 1, 1/2, 1/3],
    [5, 3, 2, 1, 1/2],
    [6, 4, 3, 2, 1]
])

# A4: Customization (C4)
A4 = np.array([
    [1, 2, 4, 3, 5],
    [1/2, 1, 3, 2, 4],
    [1/4, 1/3, 1, 1/2, 2],
    [1/3, 1/2, 2, 1, 3],
    [1/5, 1/4, 1/2, 1/3, 1]
])

def calculate_priority_vector(matrix):
    """
    Calculates eigenvector corresponding to max eigenvalue.
    Returns normalized priority vector and consistency ratio.
    """
    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(np.real(eigvals))
    max_eigval = np.real(eigvals[max_idx])
    eigvec = np.real(eigvecs[:, max_idx])
    
    # Normalize (ensure sum is 1 and values are positive)
    eigvec = eigvec / np.sum(eigvec)
    if np.any(eigvec < 0): # Handle potential negative eigenvector direction
        eigvec = -eigvec
        
    # CI and CR
    ci = (max_eigval - n) / (n - 1) if n > 1 else 0
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    ri = ri_dict.get(n, 1.49) 
    cr = ci / ri if ri != 0 else 0
    
    return eigvec, cr

def get_consistency_report(matrix, weights):
    """
    Identifies inconsistent judgments.
    Returns list of dictionaries containing details about inconsistencies.
    """
    n = matrix.shape[0]
    report = []
    
    # Threshold for highlighting (log deviation)
    # log(9) is approx 2.2. log(3) is 1.1.
    # A significant deviation might be > 0.5 (approx factor of 1.6)
    
    for i in range(n):
        for j in range(n):
            if i < j:
                actual = matrix[i, j]
                ideal = weights[i] / weights[j]
                
                # We use log difference to measure scale-independent deviation
                # Avoid log(0) if weights are 0 (shouldn't happen with valid eigenvector)
                if actual <= 0 or ideal <= 0:
                    continue
                    
                diff = abs(np.log(actual) - np.log(ideal))
                
                # Heuristic: if diff is significant, report it
                # Factor of 2 difference -> ln(2) ~= 0.69
                if diff > 0.5: 
                    report.append({
                        'row': i,
                        'col': j,
                        'actual': actual,
                        'ideal': ideal,
                        'diff': diff,
                        'message': f"Comparable {CRITERIA[i+1].split(':')[0]} vs {CRITERIA[j+1].split(':')[0]}: Value {actual:.2f} differs from consistent value {ideal:.2f}"
                    })
    
    # Sort by severity
    report.sort(key=lambda x: x['diff'], reverse=True)
    return report

def get_smart_fill_value(matrix, row, col):
    """
    Tries to infer matrix[row, col] based on other values using transitivity.
    M[i,j] = M[i,k] * M[k,j]
    Returns inferred value or None.
    """
    n = matrix.shape[0]
    possible_values = []
    
    for k in range(n):
        if k == row or k == col:
            continue
            
        # Check if we have values for (row, k) and (k, col)
        # Note: matrix is initialized with 0 for empty in our new logic
        v_rk = matrix[row, k]
        v_kc = matrix[k, col]
        
        if v_rk > 0 and v_kc > 0:
            inferred = v_rk * v_kc
            possible_values.append(inferred)
            
    if possible_values:
        # Return geometric mean of all inferences for stability
        # geometric mean = exp(mean(log(values)))
        log_sum = sum(np.log(v) for v in possible_values)
        geom_mean = np.exp(log_sum / len(possible_values))
        return geom_mean
        
    return None

# Pre-calculate local priorities for alternatives
# We use the EXACT values from the paper to ensure alignment with the framework specifications.
# Re-calculating from matrices might introduce slight deviations due to floating point or algorithm differences.
LOCAL_PRIORITIES = {
    1: np.array([0.38, 0.27, 0.08, 0.14, 0.13]), # Accuracy
    2: np.array([0.42, 0.30, 0.08, 0.05, 0.15]), # Interpretability
    3: np.array([0.05, 0.10, 0.17, 0.27, 0.41]), # Development Cost
    4: np.array([0.41, 0.27, 0.09, 0.16, 0.07])  # Customization
}

def calculate_global_scores(criteria_matrix):
    """
    criteria_matrix: 4x4 numpy array (user input)
    Returns:
        weights: 4x1 vector
        cr: consistency ratio
        scores: dict {alt_idx: score}
    """
    weights, cr = calculate_priority_vector(criteria_matrix)
    
    # Global Aggregation
    S = np.zeros(5)
    for i in range(4): # For each criterion C1..C4
        # weights[i] is weight for Criterion i+1
        # LOCAL_PRIORITIES[i+1] is score vector for alternatives under Criterion i+1
        S += weights[i] * LOCAL_PRIORITIES[i+1]
        
    return weights, cr, S
