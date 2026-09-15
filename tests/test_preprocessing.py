from app.preprocessing import preprocess

def test_preprocess_basic():
    # Test 1: Punctuation removal dan lowercase
    assert preprocess("KAIN PERCA BATIK!!!") == ["kain", "perca", "batik"]

def test_preprocess_numbers_and_units():
    # Test 2: Angka dan kondisi kata tidak hilang
    tokens = preprocess("Kain perca batik, 5 KG, kondisi bersih.")
    expected = ["kain", "perca", "batik", "5", "kg", "kondisi", "bersih"]
    assert tokens == expected

def test_preprocess_whitespace():
    # Test 3: Normalisasi spasi berlebih
    assert preprocess(" kain   perca  batik ") == ["kain", "perca", "batik"]

def test_preprocess_punctuation_separator():
    # Test 4: Separator konsisten dihilangkan
    assert preprocess("kain-perca/batik") == ["kain", "perca", "batik"]

def test_preprocess_attached_units():
    # Test 5: Ekstraksi angka dan huruf yang menempel
    tokens = preprocess("KAIN PERCA GRATIS 10KG")
    expected = ["kain", "perca", "gratis", "10", "kg"]
    assert tokens == expected

def test_sastrawi_optional_stopword():
    # Memastikan Sastrawi dapat membuang stopword jika diaktifkan
    text = "kain perca yang ada di gudang"
    
    # Default OFF
    assert preprocess(text) == ["kain", "perca", "yang", "ada", "di", "gudang"]
    
    # Stopword ON
    tokens = preprocess(text, use_stopwords=True)
    assert "yang" not in tokens
    assert "di" not in tokens
    assert "ada" not in tokens
    assert tokens == ["kain", "perca", "gudang"]

def test_sastrawi_optional_stemming():
    # Memastikan Sastrawi dapat melakukan stemming jika diaktifkan
    text = "potongan kain berwarna"
    
    # Default OFF
    assert preprocess(text) == ["potongan", "kain", "berwarna"]
    
    # Stemming ON
    tokens = preprocess(text, use_stemming=True)
    assert tokens == ["potong", "kain", "warna"]