import pytest
import math
from app.cosine_similarity import calculate_cosine_similarity

def test_identical_vectors():
    # TEST 1 — Identical vectors
    A = [1.0, 2.0, 3.0]
    B = [1.0, 2.0, 3.0]
    assert calculate_cosine_similarity(A, B) == pytest.approx(1.0)

def test_same_direction():
    # TEST 2 — Same direction
    A = [1.0, 2.0, 3.0]
    B = [2.0, 4.0, 6.0]
    assert calculate_cosine_similarity(A, B) == pytest.approx(1.0)

def test_orthogonal_vectors():
    # TEST 3 — Orthogonal vectors
    A = [1.0, 0.0]
    B = [0.0, 1.0]
    assert calculate_cosine_similarity(A, B) == pytest.approx(0.0)

def test_opposite_direction():
    # TEST 4 — Opposite direction
    A = [1.0, 0.0]
    B = [-1.0, 0.0]
    assert calculate_cosine_similarity(A, B) == pytest.approx(-1.0)

def test_general_vectors():
    # TEST 5 — General vectors
    # A = [1.0, 2.0], B = [2.0, 3.0]
    # Dot = 2 + 6 = 8, Norm A = sqrt(5), Norm B = sqrt(13)
    # Cosine = 8 / sqrt(65) ≈ 0.99227787
    A = [1.0, 2.0]
    B = [2.0, 3.0]
    expected = 8.0 / math.sqrt(65)
    assert calculate_cosine_similarity(A, B) == pytest.approx(expected)

def test_dimension_mismatch():
    # TEST 6 — Dimension mismatch
    A = [1.0, 2.0, 3.0]
    B = [1.0, 2.0]
    with pytest.raises(ValueError, match="Dimensi vektor tidak sama"):
        calculate_cosine_similarity(A, B)

def test_empty_vectors():
    # TEST 7 — Empty vectors
    A = []
    B = []
    with pytest.raises(ValueError, match="Vektor tidak boleh kosong"):
        calculate_cosine_similarity(A, B)

def test_zero_vector():
    # TEST 8 — Zero vector
    A = [0.0, 0.0, 0.0]
    B = [1.0, 2.0, 3.0]
    # Menghindari division by zero, hasilnya ditetapkan 0.0 secara behavior program
    assert calculate_cosine_similarity(A, B) == 0.0

def test_symmetry():
    # TEST 9 — Symmetry
    A = [1.5, -2.5, 3.1]
    B = [0.5, 1.1, -1.9]
    assert calculate_cosine_similarity(A, B) == calculate_cosine_similarity(B, A)

def test_score_range():
    # TEST 10 — Score range
    A = [12.3, -4.5, 6.7]
    B = [-8.9, 10.11, -12.13]
    score = calculate_cosine_similarity(A, B)
    assert -1.0 <= score <= 1.0