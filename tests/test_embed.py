from fastapi.testclient import TestClient
from app.main import app

def test_embed_valid_text():
    # Menggunakan 'with' agar lifespan (startup/shutdown) dieksekusi
    with TestClient(app) as client:
        response = client.post("/v1/embed", json={
            "text": "Kain perca batik campur warna, ukuran variatif, kondisi bersih"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "embedding" in data
        assert len(data["embedding"]) == 384
        assert isinstance(data["embedding"][0], float)
        assert data["model"] == "minilm-l12-v2"
        assert data["dim"] == 384

def test_embed_empty_text():
    with TestClient(app) as client:
        response = client.post("/v1/embed", json={"text": ""})
        # Harus memicu HTTP 422 Unprocessable Entity dari Pydantic (min_length=1)
        assert response.status_code == 422

def test_embed_missing_text():
    with TestClient(app) as client:
        response = client.post("/v1/embed", json={})
        # HTTP 422 karena field text hilang
        assert response.status_code == 422