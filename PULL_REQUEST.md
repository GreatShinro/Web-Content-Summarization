# Pull Request: Intelligent Web Content Summarization System

## Summary

Implements a full-stack web content summarization application (`app.py`) built with Streamlit. The system fetches web pages, summarizes their content using extractive or AI-based abstractive methods, and reads summaries aloud via text-to-speech — designed for enhanced accessibility.

---

## Changes

- **`app.py`** — Complete application (single-file):
  - Web scraper using `requests` + `BeautifulSoup4`
  - Extractive summarizer (TextRank-inspired, via NLTK)
  - Abstractive summarizer (HuggingFace T5-small, chunked for long texts)
  - Hybrid mode (extract → abstract)
  - Text-to-Speech output via gTTS (6 languages)
  - SQLite persistence for session history and user feedback
  - Metrics: compression ratio, readability, average sentence length
  - Dark-themed, WCAG-compliant Streamlit UI

---

## How to Test

```bash
pip install streamlit requests beautifulsoup4 transformers torch nltk gTTS
streamlit run app.py
```

Open `http://localhost:8501`, paste a URL, and click **Summarize →**.

---

## Checklist

- [x] Extractive summarization works on arbitrary URLs
- [x] Abstractive (T5-small) handles long texts via chunking
- [x] Hybrid mode produces higher-quality summaries
- [x] TTS audio generated and playable in-browser
- [x] History persisted across sessions (SQLite)
- [x] Download buttons for summary text, full report, and audio
- [x] Feedback (star rating + comment) saved to DB

---

## Notes

- T5-small is downloaded on first run (~240 MB); subsequent runs use the cache.
- JavaScript-rendered pages and paywalled content may return limited text.
- Timeout is configurable (5–30 s) in the sidebar.

---

**Author:** Abdurrahman Abubakar Said (CSA/2023/30777)  
**Supervisor:** Dr. Umar Iliyasu  
**Department:** Computer Science, Federal University Dutsin-Ma  
**Session:** 2024/2025
