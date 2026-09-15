import numpy as np

def calculate_cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Menghitung cosine similarity antara dua buah vektor.
    
    :param vector_a: Vektor pertama (list of floats)
    :param vector_b: Vektor kedua (list of floats)
    :return: Float score dari rentang -1.0 hingga 1.0 (0.0 jika salah satu zero vector)
    """
    # Edge Case: Empty vectors
    if not vector_a or not vector_b:
        raise ValueError("Vektor tidak boleh kosong.")

    # Edge Case: Dimension mismatch
    if len(vector_a) != len(vector_b):
        raise ValueError(f"Dimensi vektor tidak sama: {len(vector_a)} != {len(vector_b)}")

    # Menggunakan NumPy untuk operasi matriks
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Edge Case: Zero vector (menghindari division by zero)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # (A · B) / (||A|| * ||B||)
    similarity = np.dot(a, b) / (norm_a * norm_b)
    
    # Clip hasil ke range matematis aman [-1.0, 1.0] untuk menghindari floating point inaccuracy (misal 1.0000000000000002)
    return float(np.clip(similarity, -1.0, 1.0))