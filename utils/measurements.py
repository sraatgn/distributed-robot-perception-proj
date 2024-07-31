import numpy as np

def measurement_model(xi, xj):
    """
    Measurement model hij(xi, xj) for the relative measurement between robots i and j.
    
    Parameters:
    xi (np.array): State vector of robot i.
    xj (np.array): State vector of robot j.
    
    Returns:
    np.array: Measurement model output.
    """
    # simple measurement model: Euclidean distance between robots
    return np.linalg.norm(xi - xj)

def simulate_measurement(xi, xj, Ri):
    """
    Simulates the relative measurement zij(k + 1) collected by robot i from robot j.
    
    Parameters:
    xi (np.array): State vector of robot i.
    xj (np.array): State vector of robot j.
    Ri (np.array): Covariance matrix of the measurement noise for robot i.
    
    Returns:
    np.array: Relative measurement zij(k + 1).
    """
    # Get the measurement model output
    hij_output = measurement_model(xi, xj)
    
    # Simulate the measurement noise
    # independent zero-mean white Gaussian
    ni = np.random.multivariate_normal(np.zeros(Ri.shape[0]), Ri)
    
    # Compute the relative measurement
    zij = hij_output + ni
    
    return zij