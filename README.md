# Malay → Indonesian Speech Translation System

Sistem terjemahan ucapan (*speech translation*) ujung ke ujung (end-to-end) dari bahasa Melayu ke bahasa Indonesia, menggunakan **Whisper** (ASR) dan **M2M100** (MT) sebagai tulang punggung model.

---

## Gambaran Umum

Pipeline ini menggabungkan dua pendekatan inferensi:

| Mode | Alur |
|---|---|
| **E2E (End-to-End)** | Audio Melayu → Whisper (fine-tuned) → Teks Indonesia |
| **Cascade Pipeline** | Audio Melayu → Whisper ASR → Teks Melayu → M2M100 MT → Teks Indonesia |

Kedua mode dievaluasi menggunakan skor **BLEU** pada subset data uji.

---

## Arsitektur

```
Audio Input (Melayu)
        │
        ▼
┌─────────────────┐
│  Whisper-small  │  ← fine-tuned multitask (ASR + ST)
│  (ASR / ST)     │
└────────┬────────┘
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
Teks Melayu          Teks Indonesia
(mode ASR)           (mode E2E ST)
    │
    ▼
┌──────────────────┐
│  M2M100 (418M)   │  ← digunakan pada cascade pipeline
│  Melayu → Indo   │
└──────────────────┘
    │
    ▼
Teks Indonesia
(mode Cascade)
```

---

## Prasyarat

- Python 3.8+
- CUDA (opsional, namun sangat disarankan)

### Instalasi Dependensi

```bash
pip install transformers datasets datacollective opustools torchaudio librosa sentencepiece sacrebleu torch
```

---

## Dataset

| Dataset | Kegunaan | Sumber |
|---|---|---|
| `mesolitica/nusantara-audiobook-annotated` | Data audio + transkrip Melayu | Hugging Face |
| `DigitalLearningGmbH/tatoeba_mt_parquet` (zsm_Latn-ind) | Pasangan kalimat bitext Melayu-Indonesia | Hugging Face |

Data sintetik dihasilkan secara otomatis dengan menerjemahkan transkrip Melayu menggunakan M2M100.

---

## Alur Pelatihan

### 1. Load & Preprocessing Data
- Audio di-*resample* ke 16 kHz
- Transkrip dinormalisasi (lowercase, strip)
- Kalimat terlalu pendek (<1 detik) dibuang

### 2. Pembersihan Bitext
- Filter panjang token: 3–50 token per kalimat
- Filter rasio panjang sumber/target: 0.5–2.0

### 3. Load Model
- **MT Model:** `facebook/m2m100_418M` (Melayu → Indonesia)
- **ASR Model:** `openai/whisper-small`

### 4. Pembuatan Data Sintetik
- Setiap sampel audio diterjemahkan teks transkripnya menggunakan M2M100
- Menghasilkan triplet: `(audio, text_ms, text_id)`

### 5. Dataset Multitask
- Setiap sampel menghasilkan dua entri pelatihan:
  - `task=asr` → target: teks Melayu
  - `task=st`  → target: teks Indonesia

### 6. Pelatihan Whisper
- **Optimizer:** AdamW (`lr=1e-5`)
- **Batch size:** 4
- **Epochs:** 3
- **Strategi encoder freeze:** Encoder dibekukan di epoch pertama, dibuka di epoch berikutnya
- **Balanced sampling:** 50/50 ASR dan ST per batch

---

## Hyperparameter

| Parameter | Nilai |
|---|---|
| Learning rate | `1e-5` |
| Batch size | `4` |
| Num epochs | `3` |
| Freeze encoder epochs | `1` |
| Max source token length | `128` |
| Max target token length | `128` |
| Max synthetic samples | `200` |
| Beam search (MT) | `4` |

---

## Inferensi

### End-to-End Speech Translation
```python
result = infer_e2e_st(audio_array, sr=16000)
print(result)  # → Teks Indonesia
```

### Cascade Pipeline (ASR → MT)
```python
result = infer_pipeline_baseline(audio_array, sr=16000)
print(result["asr_output"])   # → Teks Melayu
print(result["mt_output"])    # → Teks Indonesia
```

### Perbandingan Kedua Mode
```python
results = compare_inference(audio_array, sr=16000)
print(results["e2e_st_output"])      # E2E
print(results["pipeline_mt_output"]) # Cascade
```

---

## Evaluasi

Evaluasi dilakukan dengan **BLEU score** (tokenizer `flores200`) pada hingga 50 sampel:

```
===================================================
 E2E Speech Translation BLEU : xx.xx
 Pipeline Cascade BLEU       : xx.xx
===================================================
```

---

## Struktur Direktori

```
.
├── data/
│   ├── mesolitica-nusantara-annotated/   # Cache audio dataset
│   └── tatoeba_mt_parquet/               # Cache bitext dataset
├── models/
│   ├── whisper_multitask_epoch1/         # Checkpoint epoch 1
│   ├── whisper_multitask_epoch2/         # Checkpoint epoch 2
│   └── whisper_multitask_epoch3/         # Checkpoint epoch 3 (final)
└── malay_indonesian_speech_translation.ipynb
```

---

## Catatan & Rekomendasi

| Topik | Catatan |
|---|---|
| **Model pretrained** | Selalu mulai dari `openai/whisper-small` dan `facebook/m2m100_418M` |
| **Label sintetik** | Target `text_id` dari MT bersifat *noisy*. Data bitext nyata (OpenSubtitles, CCMatrix) akan meningkatkan kualitas secara signifikan |
| **Keselarasan data** | `audio`, `text_ms`, dan `text_id` harus tetap sejajar di semua transformasi |
| **Batch size** | Pertahankan 4–8 pada single-GPU; gradient accumulation dapat ditambahkan untuk batch efektif yang lebih besar |
| **Multitask balance** | Mixing 50/50 ASR/ST mencegah model lupa transkripsi Melayu saat belajar terjemahan Indonesia |
| **Encoder freeze** | Bekukan encoder Whisper di epoch pertama untuk menstabilkan decoder sebelum fine-tuning bersama |

### Langkah Selanjutnya
1. Ganti data bitext *placeholder* dengan data OpenSubtitles MS-ID yang nyata
2. Skalakan ke split train penuh Common Voice
3. Tambahkan metrik **WER** untuk evaluasi task ASR

---

## Referensi Model

- [openai/whisper-small](https://huggingface.co/openai/whisper-small)
- [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M)
- [mesolitica/nusantara-audiobook-annotated](https://huggingface.co/datasets/mesolitica/nusantara-audiobook-annotated)
- [Tatoeba MT Parquet](https://huggingface.co/datasets/DigitalLearningGmbH/tatoeba_mt_parquet)
