from typing import Any
from rank_bm25 import BM25Okapi
from app.preprocessing import preprocess

def rank_candidates_bm25(
    query_text: str,
    candidates: list[dict[str, Any]],
    top_k: int = 10
) -> list[dict[str, Any]]:
    """
    Melakukan lexical ranking pada kandidat menggunakan BM25Okapi.
    
    :param query_text: Teks kebutuhan pencarian dari user
    :param candidates: List dict berisi setidaknya 'candidate_id' dan 'candidate_text'
    :param top_k: Batas jumlah kandidat teratas yang diambil (default 10)
    :return: List dict terurut descending berdasarkan 'bm25_score'
    """
    # Validation & Edge Cases Handling
    if not candidates or not query_text or top_k <= 0:
        return []

    # Preprocessing query
    query_tokens = preprocess(query_text)
    if not query_tokens:
        return []

    # Preprocessing seluruh kandidat menggunakan fungsi dari Phase 4
    corpus_tokens = [
        preprocess(item.get("candidate_text", ""))
        for item in candidates
    ]

    # Inisialisasi BM25Okapi Engine
    bm25 = BM25Okapi(corpus_tokens)

    # Kalkulasi skor relevansi leksikal
    scores = bm25.get_scores(query_tokens)

    # Menjaga integritas hubungan candidate_id, candidate_text, dan bm25_score
    results = []
    for idx, item in enumerate(candidates):
        results.append({
            "candidate_id": item.get("candidate_id"),
            "candidate_text": item.get("candidate_text", ""),
            "bm25_score": float(scores[idx])
        })

    # Urutkan secara descending berdasarkan score BM25
    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    # Ambil top_k (jika candidate < top_k, otomatis mengembalikan seluruh candidate)
    return results[:top_k]