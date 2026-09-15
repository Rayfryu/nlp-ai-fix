# demo_matching.py
from app.bm25 import rank_candidates_bm25

def run_demo():
    user_query = "Butuh kain perca batik minimal 5kg, kondisi bersih"
    
    kandidat_material = [
        {"candidate_id": "MAT-01", "candidate_text": "Kain perca batik campur, 10kg, bersih"},
        {"candidate_id": "MAT-02", "candidate_text": "Kardus bekas berbagai ukuran"},
        {"candidate_id": "MAT-03", "candidate_text": "Kain katun polos 5kg"},
        {"candidate_id": "MAT-04", "candidate_text": "Sisa potongan batik solo bersih"}
    ]

    print(f"\n🔍 PERMINTAAN PENGGUNA: \"{user_query}\"\n")
    print("📊 MEMPROSES ALGORITMA BM25 LEKSIKAL...\n")
    
    hasil = rank_candidates_bm25(user_query, kandidat_material, top_k=3)
    
    print("RECOMMENDED MATCHES (TOP 3):")
    for i, item in enumerate(hasil, 1):
        print(f"{i}. ID: {item['candidate_id']} | Teks: \"{item['candidate_text']}\" | Skor: {item['bm25_score']:.2f}")

if __name__ == "__main__":
    run_demo()