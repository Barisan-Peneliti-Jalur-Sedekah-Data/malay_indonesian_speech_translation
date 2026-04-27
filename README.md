# Malay → Indonesian Speech Translation System

**Kelompok 4 — Progress 1**

Sistem *speech translation* end-to-end dari bahasa Melayu ke bahasa Indonesia menggunakan pipeline: **Common Voice (MS) + Synthetic MT Data + Whisper + M2M100**.

---

## Deskripsi Proyek

Proyek ini membangun sistem penerjemahan ucapan (*speech translation*) dari bahasa Melayu ke bahasa Indonesia dengan dua pendekatan:

1. **End-to-End (E2E)** — Whisper yang di-*fine-tune* langsung menghasilkan teks Indonesia dari audio Melayu.
2. **Cascade Pipeline** — ASR (Whisper → teks Melayu) → MT (M2M100 → teks Indonesia).

Kedua pendekatan dievaluasi dan dibandingkan menggunakan metrik **BLEU** dan **ChrF**.

---

## Arsitektur Pipeline

```
Audio (Melayu)
      │
      ▼
┌─────────────┐
│   Whisper   │  ← openai/whisper-small (fine-tuned multitask)
│   (ASR/ST)  │
└─────────────┘
      │
   ┌──┴──┐
   │     │
   ▼     ▼
[ASR]  [ST - E2E]
Teks    Teks ID
Melayu  langsung
   │
   ▼
┌──────────────┐
│   M2M100    │  ← facebook/m2m100_418M
│  (MT: ms→id)│
└──────────────┘
   │
   ▼
Teks Indonesia
```

---

##  Dependencies

Install semua dependensi dengan perintah berikut:

```bash
pip install transformers datasets datacollective opustools torchaudio librosa sentencepiece sacrebleu torch
```

| Library | Versi | Kegunaan |
|---|---|---|
| `transformers` | latest | Whisper & M2M100 |
| `datasets` | latest | Load & proses dataset |
| `torchaudio` | latest | Audio processing |
| `librosa` | latest | Resampling audio |
| `sentencepiece` | latest | Tokenizer M2M100 |
| `sacrebleu` | latest | Evaluasi BLEU & ChrF |
| `torch` | latest | Backend deep learning |

---

## Struktur Direktori

```
project/
├── data/
│   ├── mesolitica-nusantara-annotated/   # Dataset audio Melayu
│   └── tatoeba_mt_parquet/               # Dataset bitext MS-ID
├── models/
│   ├── whisper_multitask_epoch1/         # Checkpoint epoch 1
│   ├── whisper_multitask_epoch2/         # Checkpoint epoch 2
│   └── whisper_multitask_epoch3/         # Checkpoint epoch 3
├── loss_curve.png                        # Grafik training loss
└── kelompok_4_progress_1.ipynb           # Notebook utama
```

---

## Dataset

| Dataset | Sumber | Kegunaan |
|---|---|---|
| `mesolitica/nusantara-audiobook-annotated` | HuggingFace | Audio + transkripsi Melayu |
| `DigitalLearningGmbH/tatoeba_mt_parquet` (zsm_Latn-ind) | HuggingFace | Pasangan kalimat MS-ID |

### Preprocessing Audio
- Sampling rate: **16.000 Hz**
- Durasi minimum: **1 detik**
- Format kolom: `audio`, `text_ms`, `duration`

### Filtering Bitext
- Panjang token: **3–50 token**
- Rasio panjang src/tgt: **0.5–2.0**

---

## Alur Notebook (Section by Section)

| Section | Judul | Deskripsi |
|---|---|---|
| 0 | Setup | Install dependensi, import library, setup device & direktori |
| 1 | Load & Preprocess Data | Load dataset audio Melayu + bitext MS-ID |
| 2 | Clean Bitext Data | Filter pasangan kalimat berdasarkan panjang & rasio |
| 3 | Load MT Model | Load M2M100 (418M) untuk translasi Melayu → Indonesia |
| 4 | Load ASR Model | Load Whisper-small untuk ASR |
| 5 | Generate Synthetic ST Data | Terjemahkan transkripsi audio menggunakan M2M100 (200 sampel) |
| 6 | Build Multitask Dataset | Gabungkan data ASR + ST (interleaved 50/50) |
| 7 | Prepare Whisper Inputs | Ekstraksi fitur log-mel + tokenisasi label |
| 8 | Train End-to-End Model | Fine-tuning Whisper dengan multitask (ASR + ST) |
| 9 | Inference Functions | Fungsi inferensi E2E dan cascade pipeline |
| 10 | Evaluation | Evaluasi BLEU & ChrF + analisis kualitatif |

---

## Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Model | `openai/whisper-small` |
| Optimizer | AdamW |
| Learning Rate | `1e-5` |
| Batch Size | `4` |
| Epochs | `3` |
| Encoder Freeze | Epoch 1 (frozen), Epoch 2–3 (unfrozen) |
| Max Target Length | `128` token |
| Gradient Clipping | `max_norm=1.0` |

### Hasil Training Loss

| Epoch | Avg Loss |
|---|---|
| 1 | 1.6005 |
| 2 | 0.8677 |
| 3 | 0.3792 |

> Loss turun secara konsisten dari **1.60 → 0.38** selama 3 epoch.

---

## Evaluasi

Evaluasi dilakukan pada **50 sampel** menggunakan:
- **BLEU** (tokenizer: `flores200`)
- **ChrF** (word order: 2)

Dua sistem yang dibandingkan:
- **E2E ST**: Whisper fine-tuned langsung menghasilkan teks Indonesia
- **Cascade Pipeline**: Whisper ASR → M2M100 MT

---

## Catatan Pengembangan

| Topik | Keterangan |
|---|---|
| **Pretrained models** | Selalu mulai dari `openai/whisper-small` dan `facebook/m2m100_418M`; jangan latih dari awal. |
| **Label sintetis** | Target `text_id` dihasilkan MT sehingga mengandung noise. Dataset bitext nyata (OpenSubtitles, CCMatrix) akan meningkatkan kualitas secara signifikan. |
| **Keselarasan data** | Kolom `audio`, `text_ms`, dan `text_id` harus tetap selaras di seluruh transformasi. |
| **Batch size** | Tetap 4–8 pada single-GPU; gradient accumulation dapat ditambahkan untuk batch efektif yang lebih besar. |
| **Multitask balance** | Pencampuran ASR/ST 50/50 mencegah model melupakan transkripsi Melayu saat belajar translasi Indonesia. |
| **Encoder freeze** | Bekukan encoder Whisper pada epoch pertama untuk menstabilkan decoder sebelum fine-tuning bersama. |

---

## Next Steps

1. Ganti data bitext placeholder dengan data **OpenSubtitles MS-ID** yang nyata.
2. Skalakan ke **full split train** Common Voice Melayu.
3. Tambahkan metrik **WER (Word Error Rate)** untuk evaluasi task ASR.

---

## Tim

**Kelompok 4** — Progress Report 1
