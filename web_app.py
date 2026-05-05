"""
Minimal web interface for the preset chatbot.
Run with: python web_app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string

from WineBot import WineChatbot

load_dotenv()

# Same config as WineBot __main__
DB_DESTINATION_PATH = "fnh330_db/"
USE_SEMANTIC_CHUNKING = False
base_data_path = os.path.join("wine_data")
data_dirs = [
    os.path.join(base_data_path, "Lectures"),
    os.path.join(base_data_path, "Notes"),
    os.path.join(base_data_path, "video_transcripts"),
]
  
app = Flask(__name__)
_chatbot = None


def get_chatbot():
    global _chatbot
    if _chatbot is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment")
        _chatbot = WineChatbot(
            llm_model_name="gemini-2.5-flash",
            embedding_model_name="gemini-embedding-001",
            api_key=api_key,
            collection_name="fnh330_wine_course_gemini",
            db_path=DB_DESTINATION_PATH,
            chunking_mode="semantic" if USE_SEMANTIC_CHUNKING else "recursive",
        )
    return _chatbot


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    image_data = data.get("image_data")
    image_mime_type = data.get("image_mime_type")

    # Debug stuff
    print(f"[/ask] prompt: {prompt[:50]!r}")
    print(f"[/ask] image_data present: {image_data is not None}")
    print(f"[/ask] image_data length: {len(image_data) if image_data else 0}")
    print(f"[/ask] image_mime_type: {image_mime_type}")
    
    if not prompt and not image_data:
        return jsonify({"error": "prompt or image is required"}), 400
    try:
        bot = get_chatbot()
        response = bot.run(prompt, image_data=image_data, image_mime_type=image_mime_type)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Knowledge Base Assistant</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #1a1b1e;
      --surface: #25262b;
      --border: #2c2e33;
      --text: #c1c2c5;
      --muted: #868e96;
      --accent: #5c7cfa;
      --accent-hover: #748ffc;
      --prompt-text: var(--text);
      --output-text: #e8e9eb;
    }
    [data-theme="light"] {
      --bg: #f1f3f5;
      --surface: #fff;
      --border: #dee2e6;
      --text: #212529;
      --muted: #868e96;
      --accent: #4c6ef5;
      --accent-hover: #364fc7;
      --prompt-text: #e4e6e8;
      --output-text: #111213;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 1rem;
    }
    .container {
      width: 100%;
      max-width: 42rem;
    }
    h1 {
      font-size: 1rem;
      font-weight: 600;
      color: var(--muted);
      margin: 0 0 1rem 0;
    }
    .input-row { margin-bottom: 1rem; }
    #prompt {
      width: 100%;
      padding: 0.6rem 0.75rem;
      background: var(--bg);
      border: none;
      border-radius: 6px;
      color: var(--prompt-text);
      font-size: 0.95rem;
      resize: none;
      min-height: 44px;
      max-height: 120px;
    }
    #prompt:focus {
      outline: none;
    }
    #prompt::placeholder { color: var(--muted); }
    #output {
      background: var(--bg);
      border: none;
      border-radius: 6px;
      padding: 1rem;
      min-height: 6rem;
      word-break: break-word;
      font-size: 0.9rem;
      line-height: 1.5;
      color: var(--output-text);
    }
    #output.empty { color: var(--muted); }
    #output.error { color: #fa5252; }
    #output p { margin: 0 0 0.75em 0; }
    #output p:last-child { margin-bottom: 0; }
    #output ul, #output ol { margin: 0.5em 0; padding-left: 1.5em; }
    #output code { background: var(--border); padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }
    #output pre { background: var(--border); padding: 0.75rem; border-radius: 6px; overflow-x: auto; margin: 0.75em 0; }
    #output pre code { background: none; padding: 0; }
    .status { font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem; }
    .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; gap: 0.5rem; }
    .header h1 { margin: 0; }
    #theme-toggle {
      padding: 0.35rem 0.65rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      font-size: 0.85rem;
      cursor: pointer;
    }
    #theme-toggle:hover { border-color: var(--accent); }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Knowledge Base Assistant</h1>
      <button type="button" id="theme-toggle" aria-label="Toggle light/dark mode">Light</button>
    </div>
    <div class="input-row">
      <textarea id="prompt" rows="1" placeholder="" autofocus spellcheck="false"></textarea>
      <div id="image-preview-container" style="display: none; margin-top: 0.5rem; position: relative; width: fit-content;">
        <img id="image-preview" style="max-height: 100px; border-radius: 4px; border: 1px solid var(--border);" />
        <button id="remove-image" type="button" style="position: absolute; top: -5px; right: -5px; background: red; color: white; border: none; border-radius: 50%; width: 20px; height: 20px; cursor: pointer; font-size: 12px; line-height: 1;">&times;</button>
      </div>
    </div>
    <div id="output" class="empty">Ask a question about the loaded knowledge base.</div>
    <div id="status" class="status" aria-live="polite"></div>
  </div>
  <script>
    const promptEl = document.getElementById('prompt');
    const outputEl = document.getElementById('output');
    const statusEl = document.getElementById('status');
    const themeToggle = document.getElementById('theme-toggle');
    const imgPreviewContainer = document.getElementById('image-preview-container');
    const imgPreview = document.getElementById('image-preview');
    const removeImgBtn = document.getElementById('remove-image');

    let currentImageData = null;
    let currentImageMimeType = null;

    removeImgBtn.addEventListener('click', () => {
      currentImageData = null;
      currentImageMimeType = null;
      imgPreviewContainer.style.display = 'none';
      imgPreview.src = '';
    });

    promptEl.addEventListener('paste', (e) => {
      const items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (const item of items) {
        if (item.type.indexOf('image/') === 0) {
          const blob = item.getAsFile();
          const reader = new FileReader();
          reader.onload = (event) => {
            const dataUrl = event.target.result;
            // Fallback: parse MIME from data URL if item.type empty
            currentImageMimeType = item.type || dataUrl.split(';')[0].split(':')[1] || 'image/png';
            currentImageData = dataUrl.split(',')[1];
            imgPreview.src = dataUrl;
            imgPreviewContainer.style.display = 'block';
          };
          reader.readAsDataURL(blob);
          e.preventDefault();
          break;
        }
      }
    });

    const THEME_KEY = 'wino-theme';
    function getTheme() { return document.documentElement.getAttribute('data-theme') || 'dark'; }
    function setTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      themeToggle.textContent = theme === 'dark' ? 'Light' : 'Dark';
      try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
    }
    function initTheme() {
      try {
        const saved = localStorage.getItem(THEME_KEY);
        if (saved === 'light') setTheme('light');
        else setTheme('dark');
      } catch (e) { setTheme('dark'); }
    }
    themeToggle.addEventListener('click', function() {
      setTheme(getTheme() === 'dark' ? 'light' : 'dark');
    });
    initTheme();

    function setStatus(msg) { statusEl.textContent = msg || ''; }
    function setBusy(busy) { promptEl.disabled = busy; }

    async function submit() {
      const prompt = promptEl.value.trim();
      if (!prompt && !currentImageData) return;
      outputEl.textContent = '';
      outputEl.classList.remove('empty', 'error');
      setBusy(true);
      setStatus('🍇🤔🍷🥴🥂');
      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            prompt,
            image_data: currentImageData,
            image_mime_type: currentImageMimeType
          })
        });
        const data = await res.json();
        if (!res.ok) {
          outputEl.textContent = data.error || 'Request failed';
          outputEl.classList.add('error');
          setStatus('');
          return;
        }
        const html = marked.parse(data.response || '(No response)');
        outputEl.innerHTML = html;
        promptEl.value = '';
        removeImgBtn.click();
        setStatus('');
      } catch (err) {
        outputEl.textContent = err.message || 'Network error';
        outputEl.classList.add('error');
        setStatus('');
      } finally {
        setBusy(false);
      }
    }

    promptEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
