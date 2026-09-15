from app.bm25 import rank_candidates_bm25

def test_bm25_basic_relevance():
    query = "kain perca batik"
    candidates = [
        {"candidate_id": "A", "candidate_text": "kain perca batik bersih"},
        {"candidate_id": "B", "candidate_text": "kardus bekas"},
        {"candidate_id": "C", "candidate_text": "kain katun"}
    ]
    results = rank_candidates_bm25(query, candidates, top_k=10)
    
    assert len(results) == 3
    assert results[0]["candidate_id"] == "A"
    assert results[0]["bm25_score"] > results[1]["bm25_score"]

def test_bm25_keyword_relevance():
    query = "kain perca"
    candidates = [
        {"candidate_id": "A", "candidate_text": "kain perca batik"},
        {"candidate_id": "B", "candidate_text": "kardus bekas"},
        {"candidate_id": "C", "candidate_text": "kayu bekas"}
    ]
    results = rank_candidates_bm25(query, candidates, top_k=10)
    
    assert results[0]["candidate_id"] == "A"
    assert results[0]["bm25_score"] > 0.0

def test_bm25_multiple_relevant_candidates():
    query = "kain perca batik"
    candidates = [
        {"candidate_id": "A", "candidate_text": "kain perca batik bersih"},
        {"candidate_id": "B", "candidate_text": "kain perca batik campur"},
        {"candidate_id": "C", "candidate_text": "kain katun"},
        {"candidate_id": "D", "candidate_text": "kardus bekas kering"},
        {"candidate_id": "E", "candidate_text": "perabot kayu bekas"}
    ]
    results = rank_candidates_bm25(query, candidates, top_k=10)
    
    top_two_ids = [results[0]["candidate_id"], results[1]["candidate_id"]]
    assert "A" in top_two_ids
    assert "B" in top_two_ids
    assert results[2]["candidate_id"] == "C"

def test_bm25_top_k_limiting():
    query = "kain perca batik"
    candidates = [
        {"candidate_id": f"C{i}", "candidate_text": f"kain perca item {i}"}
        for i in range(5)
    ]
    results = rank_candidates_bm25(query, candidates, top_k=2)
    
    assert len(results) == 2

def test_bm25_candidates_less_than_top_k():
    query = "kain perca batik"
    candidates = [
        {"candidate_id": "A", "candidate_text": "kain perca batik bersih"},
        {"candidate_id": "B", "candidate_text": "kardus bekas"},
        {"candidate_id": "C", "candidate_text": "kain katun"}
    ]
    results = rank_candidates_bm25(query, candidates, top_k=10)
    
    assert len(results) == 3

def test_bm25_empty_candidates_and_query():
    # Empty candidates
    assert rank_candidates_bm25("kain", [], top_k=10) == []
    # Empty query
    assert rank_candidates_bm25("", [{"candidate_id": "A", "candidate_text": "kain"}], top_k=10) == []
    # top_k invalid
    assert rank_candidates_bm25("kain", [{"candidate_id": "A", "candidate_text": "kain"}], top_k=0) == []

def test_bm25_preprocessing_integration():
    # Menguji integrasi otomatis tokenisasi "10kg" menjadi ["10", "kg"]
    query = "kain perca 10kg"
    candidates = [
        {"candidate_id": "A", "candidate_text": "kain perca 10 kg bersih"},
        {"candidate_id": "B", "candidate_text": "kain perca 5kg"}
    ]
    results = rank_candidates_bm25(query, candidates, top_k=2)
    
    assert results[0]["candidate_id"] == "A"

def test_bm25_indonesian_informal_text():
    query = "Saya punya botol plastik bekas 10 kg"
    candidates = [
        {"candidate_id": "L001", "candidate_text": "Botol plastik bekas 10 kg"},
        {"candidate_id": "L002", "candidate_text": "Kardus bekas 20 kg"},
        {"candidate_id": "L003", "candidate_text": "Besi tua 10 kg"},
        {"candidate_id": "L004", "candidate_text": "Botol kaca bekas"},
        {"candidate_id": "L005", "candidate_text": "Sampah organik rumah tangga"}
    ]
    results = rank_candidates_bm25(query, candidates, top_k=5)
    
    # --- CETAK TEKS INFORMASI HASIL UNTUK PENGGUNA ---
    print("\n" + "="*60)
    print(f"📌 QUERY INPUT : '{query}'")
    print("="*60)
    print("🏆 HASIL RANKING BM25:")
    for rank, item in enumerate(results, 1):
        print(f"  {rank}. [{item['candidate_id']}] {item['candidate_text']}")
        print(f"     -> BM25 Score: {item['bm25_score']:.4f}")
    print("="*60)

    # Validation test tetap berjalan
    assert results[0]["candidate_id"] == "L001"