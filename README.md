# ATR Workflow GSA

A semi-automated text recognition workflow for digital editions at the Goethe- und Schiller-Archiv (Klassik Stiftung Weimar). It combines Transkribus HTR with multimodal LLMs (Claude, GPT, Gemini) and generates TEI-XML output.

> **Note:** This tool is primarily designed for use within the Goethe- und Schiller-Archiv. The two built-in project modes (*Briefe an Goethe* and *Goethes Lyrik*) include project-specific prompts and configurations. Adapting it to other projects requires changing project names, prompts, and potentially the TEI schema logic in the source code.

> This project was developed with the assistance of [Claude Code](https://claude.ai/code) and [GitHub Copilot](https://github.com/features/copilot).

---

## Requirements

- Python 3.11+
- API keys for: Anthropic, OpenAI or Google Gemini
- Transkribus credentials

---

## Installation

**1. Clone the repository and create a virtual environment:**

```bash
git clone https://github.com/KlassikStiftungWeimar/atr-workflow-gsa-public.git
cd atr-workflow-gsa
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Create a `.env` file** in the `ocr_workflow/` directory:

```env
SECRET_KEY=your-django-secret-key
ALLOWED_HOSTS=127.0.0.1

# API Keys – one set per project (RA = Briefe an Goethe, GL = Goethes Lyrik)
ANTHROPIC_KEY_RA=...
ANTHROPIC_KEY_GL=...
OPENAI_KEY_RA=...
OPENAI_KEY_GL=...
GOOGLE_KEY_RA=...
GOOGLE_KEY_GL=...

# Transkribus credentials
USERNAME_TRANSKRIBUS_RA=...
PASSWORD_TRANSKRIBUS_RA=...
USERNAME_TRANSKRIBUS_GL=...
PASSWORD_TRANSKRIBUS_GL=...
```

You only need keys for the providers you intend to use.

**4. Run migrations and start the server:**

```bash
cd ocr_workflow
python manage.py runserver
```

The app is available at `http://127.0.0.1:8000/atr_workflow/`.

> **Note (DEBUG=False):** When running with `DEBUG = False` in `settings.py` (e.g. for production), Django no longer serves static files automatically. You must collect them first:
>
> ```bash
> cd ocr_workflow
> python manage.py collectstatic
> ```
>
> This copies all static files into the `staticfiles/` directory defined by `STATIC_ROOT`. You then need a web server (e.g. nginx or whitenoise) to serve that directory.

---

## Usage

### 1. Select project mode

Toggle between **Briefe an Goethe (RA)** and **Goethes Lyrik (GL)** at the top. This controls which API credentials are used and which default TEI prompt is applied.

### 2. Upload images

Upload up to 10 document pages (JPG/PNG). All pages are processed together in one run.

### 3. Configure text recognition

**Transkribus model** – The HTR/OCR model used on Transkribus:
- *The Text Titan I* – General-purpose handwriting model
- *The Text Titan I ter* – Updated variant of Text Titan I
- *German Genius* – Optimized for German handwriting
- *Transkribus Print M1* – For printed text

**Multimodal LLM** – The language model that independently reads the image and refines the result:
- *Claude Sonnet 4.5*
- *Gemini 2.5 Pro*
- *Gemini 3 Pro Preview*
- *GPT-5.2*

**Temperature** – Controls how deterministic the LLM output is (0.0–1.0). Lower values produce more consistent, conservative results; higher values allow more variation. Does not affect Transkribus.

### 4. Start recognition

Click *Texterkennung starten*. Transkribus and the LLM run in parallel. A merged result combining both outputs is generated automatically. Progress is shown in real time.

Once complete, the **comparison view** lets you display any two versions side by side:
- *Transkribus* – Raw Transkribus output
- *Multimodales Sprachmodell* – Raw LLM output
- *Transkribus + Multimodales Sprachmodell* – Merged result
- *Eigene Version I / II* – Editable fields for manual corrections

### 5. Generate TEI-XML

Switch to the TEI panel and configure:

**Text source** – Which text version to use as the basis for TEI generation:
- *Eigene Version I / II* – Use one of the manually edited versions (recommended)
- *Transkribus + Multimodales Sprachmodell* – Use the merged result directly

**Prompt type** – The instruction given to the LLM for TEI markup:
- *Prompt Briefe an Goethe* – Pre-configured prompt for the RA project
- *Prompt Goethes Lyrik* – Pre-configured prompt for the GL project
- *Eigenes Prompt* – Enter a fully custom prompt (image and recognized text are appended automatically)

**LLM and temperature** – Same options as in the recognition step.

Click *Erstelle TEI-XML* to generate. The result is shown alongside a plain-text comparison view.