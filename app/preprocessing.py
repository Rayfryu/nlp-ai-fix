import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Lazy loading untuk Sastrawi agar tidak memperlambat startup service 
# jika fitur stemming/stopword removal sedang tidak diaktifkan.
_stemmer = None
_stopwords = None

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer

def get_stopwords():
    global _stopwords
    if _stopwords is None:
        _stopwords = StopWordRemoverFactory().get_stop_words()
    return set(_stopwords)

def normalize_text(text: str) -> str:
    """
    Mengubah teks menjadi lowercase.
    Spasi dan karakter khusus akan ditangani pada tahap tokenisasi.
    """
    return text.lower()

def tokenize(text: str) -> list[str]:
    """
    Mengekstrak urutan alfabet dan angka.
    Regex [a-z]+|[0-9]+ akan memisahkan huruf dan angka secara otomatis.
    Contoh: "10KG" menjadi ["10", "kg"], "kain-perca" menjadi ["kain", "perca"].
    Ini sangat berguna untuk menstandarkan input satuan material.
    """
    return re.findall(r'[a-z]+|[0-9]+', text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Menghapus kata-kata umum (stop words) Bahasa Indonesia.
    """
    stopwords = get_stopwords()
    return [t for t in tokens if t not in stopwords]

def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Mengubah kata menjadi kata dasarnya menggunakan Sastrawi.
    """
    stemmer = get_stemmer()
    return [stemmer.stem(t) for t in tokens]

def preprocess(text: str, use_stopwords: bool = False, use_stemming: bool = False) -> list[str]:
    """
    Fungsi utama pipeline preprocessing.
    Default: Stopwords dan Stemming OFF, sesuai spesifikasi KANA 
    agar tidak merusak konteks pencarian barang spesifik/informal.
    """
    if not isinstance(text, str) or not text.strip():
        return []
        
    text = normalize_text(text)
    tokens = tokenize(text)
    
    if use_stopwords:
        tokens = remove_stopwords(tokens)
        
    if use_stemming:
        tokens = stem_tokens(tokens)
        
    return tokens