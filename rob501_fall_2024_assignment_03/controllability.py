import numpy as np

def controllability_matrix(A, B):
    """
    Calculate the controllability matrix of the state-space system (A, B).

    Parameters:
    ----------
    A : np.ndarray
        The state matrix (n x n).
    B : np.ndarray
        The input matrix (n x m).

    Returns:
    -------
    C : np.ndarray
        The controllability matrix (n x (n * m)).
    """
    # Number of states (n)
    n = A.shape[0]

    # Initialize controllability matrix with B as the first block
    C = B

    # Append each successive power of A * B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i).dot(B)))

    return C

if __name__ == "__main__":
    # Example usage
    A = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [0, 2, -1, 0], [0, -1, 1, 1]])
    B = np.array([[1, 1], [1, 0], [1, 0], [0,0]])

    C = controllability_matrix(A, B)
    print("Controllability Matrix:\n", C)

    # Optional: Check if the system is controllable
    if np.linalg.matrix_rank(C) == A.shape[0]:
        print("The system is controllable.")
    else:
        print("The system is not controllable.")
