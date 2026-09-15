from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_ID = "minilm-l12-v2"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Memuat model {MODEL_NAME} ke memori...")
    try:
        app.state.model = SentenceTransformer(MODEL_NAME)
        app.state.model_loaded = True
        print(f"Model {MODEL_ID} berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        app.state.model = None
        app.state.model_loaded = False
    
    yield
    
    print("Menghentikan service... membersihkan memori.")
    app.state.model = None
    app.state.model_loaded = False

app = FastAPI(
    title="KANA NLP Matching Engine",
    version="0.1.0",
    lifespan=lifespan
)

# --- PYDANTIC SCHEMAS ---
class EmbedRequest(BaseModel):
    # Field text wajib ada, berupa string, dan tidak boleh string kosong ("")
    text: str = Field(..., min_length=1, description="Teks yang akan diubah menjadi vektor")

class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    dim: int

# --- ENDPOINTS ---
@app.get("/health")
def health():
    is_loaded = getattr(app.state, "model_loaded", False)
    return {
        "status": "ok" if is_loaded else "error",
        "model_loaded": is_loaded,
        "model": MODEL_ID
    }

@app.post("/v1/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest, request: Request):
    """
    Menghasilkan sentence embedding dari teks yang diberikan.
    """
    if not getattr(request.app.state, "model_loaded", False) or request.app.state.model is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "MODEL_NOT_READY", "message": "Service belum siap, model masih di-load."}}
        )

    try:
        # Melakukan inferensi menggunakan model di memory dengan normalisasi
        vector = request.app.state.model.encode(payload.text, normalize_embeddings=True).tolist()
        
        return EmbedResponse(
            embedding=vector,
            model=MODEL_ID,
            dim=len(vector)
        )
    except Exception as e:
        # Menyembunyikan stack trace dari client
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "MODEL_INFERENCE_ERROR", "message": "Terjadi kesalahan internal saat menghasilkan embedding."}}
        )