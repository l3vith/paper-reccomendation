"""Browser PDF viewer with synchronized word highlighting and speech."""

from __future__ import annotations

import base64
import html
import json
import os


PDFJS_MODULE_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs"
PDFJS_WORKER_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs"


def build_synchronized_pdf_reader(pdf_path: str, title: str, rate: int = 175) -> str:
    """Build a self-contained PDF.js reader for a local PDF.

    Speech is performed by the browser for the current page. Word boundary
    events drive the transcript highlight, so highlighting follows speech
    instead of guessing from the duration of a separately generated audio file.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    with open(pdf_path, "rb") as handle:
        pdf_base64 = base64.b64encode(handle.read()).decode("ascii")

    safe_title = html.escape(title or "Research paper")
    title_json = json.dumps(title or "Research paper")
    module_url_json = json.dumps(PDFJS_MODULE_URL)
    worker_url_json = json.dumps(PDFJS_WORKER_URL)
    rate = max(120, min(260, int(rate)))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #eef2f6; color: #172033; }}
    .reader {{ height: 820px; display: grid; grid-template-rows: auto 1fr; border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: white; }}
    .toolbar {{ display: flex; align-items: center; gap: 8px; padding: 10px 12px; border-bottom: 1px solid #dbe2ea; background: #f8fafc; flex-wrap: wrap; }}
    .title {{ font-size: 13px; font-weight: 700; margin-right: auto; max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    button, select {{ min-height: 34px; border: 1px solid #cbd5e1; border-radius: 7px; background: white; color: #172033; padding: 6px 10px; font: inherit; font-size: 12px; }}
    button {{ cursor: pointer; font-weight: 650; }}
    button:hover {{ border-color: #64748b; background: #f1f5f9; }}
    button.primary {{ background: #2563eb; border-color: #2563eb; color: white; }}
    button.primary:hover {{ background: #1d4ed8; }}
    button:disabled {{ cursor: default; opacity: .45; }}
    .page-count, .status {{ font-size: 12px; color: #64748b; }}
    .content {{ min-height: 0; display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(300px, .85fr); }}
    .pdf-pane {{ min-width: 0; overflow: auto; padding: 18px; background: #dfe5eb; text-align: center; }}
    canvas {{ display: block; max-width: 100%; height: auto !important; margin: 0 auto; background: white; box-shadow: 0 5px 24px rgba(15,23,42,.18); }}
    .transcript-pane {{ min-width: 0; display: grid; grid-template-rows: auto 1fr; border-left: 1px solid #dbe2ea; background: #fff; }}
    .transcript-header {{ padding: 13px 16px; border-bottom: 1px solid #e2e8f0; }}
    .transcript-header strong {{ display: block; font-size: 13px; }}
    .transcript-header span {{ font-size: 11px; color: #64748b; }}
    #transcript {{ overflow: auto; padding: 18px; font-family: Georgia, serif; font-size: 16px; line-height: 1.85; text-align: left; }}
    .word {{ border-radius: 3px; padding: 1px 0; transition: background-color .08s ease, color .08s ease; }}
    .word.active {{ background: #facc15; color: #111827; outline: 2px solid rgba(250,204,21,.28); }}
    .loading {{ padding: 40px 20px; color: #475569; font-size: 14px; }}
    .error {{ color: #b91c1c; }}
    @media (max-width: 760px) {{
      .reader {{ height: 1050px; }}
      .content {{ grid-template-columns: 1fr; grid-template-rows: 56% 44%; }}
      .transcript-pane {{ border-left: 0; border-top: 1px solid #dbe2ea; }}
    }}
  </style>
</head>
<body>
  <div class="reader">
    <div class="toolbar">
      <div class="title" title="{safe_title}">{safe_title}</div>
      <button id="prev" aria-label="Previous page">←</button>
      <span class="page-count">Page <b id="pageNum">1</b> / <b id="pageTotal">–</b></span>
      <button id="next" aria-label="Next page">→</button>
      <select id="voice" aria-label="Browser voice"><option value="">System voice</option></select>
      <select id="speed" aria-label="Reading speed">
        <option value="140">140 wpm</option><option value="175" selected>175 wpm</option>
        <option value="210">210 wpm</option><option value="250">250 wpm</option>
      </select>
      <button id="read" class="primary">▶ Read page</button>
      <button id="pause">Pause</button>
      <button id="stop">Stop</button>
      <span id="status" class="status">Loading PDF…</span>
    </div>
    <div class="content">
      <div class="pdf-pane"><canvas id="pdfCanvas"></canvas><div id="loading" class="loading">Rendering page…</div></div>
      <section class="transcript-pane">
        <div class="transcript-header"><strong>Live reading position</strong><span>The highlighted word follows browser speech.</span></div>
        <div id="transcript"></div>
      </section>
    </div>
  </div>
  <script type="module">
    const PDF_B64 = "{pdf_base64}";
    function base64ToBytes(b64) {{
      const binary = atob(b64);
      const length = binary.length;
      const bytes = new Uint8Array(length);
      const chunk = 1 << 15;
      for (let offset = 0; offset < length; offset += chunk) {{
        const end = Math.min(offset + chunk, length);
        for (let i = offset; i < end; i++) bytes[i] = binary.charCodeAt(i);
      }}
      return bytes;
    }}
    const pdfBytes = base64ToBytes(PDF_B64);
    const pdfjsLib = await import({module_url_json});
    pdfjsLib.GlobalWorkerOptions.workerSrc = {worker_url_json};

    const canvas = document.getElementById('pdfCanvas');
    const context = canvas.getContext('2d');
    const transcript = document.getElementById('transcript');
    const status = document.getElementById('status');
    const loading = document.getElementById('loading');
    const pageNum = document.getElementById('pageNum');
    const pageTotal = document.getElementById('pageTotal');
    const prev = document.getElementById('prev');
    const next = document.getElementById('next');
    const voiceSelect = document.getElementById('voice');
    const speedSelect = document.getElementById('speed');
    const synth = window.speechSynthesis || null;
    let pdf = null, currentPage = 1, pageText = '', wordStarts = [], activeWord = -1, utterance = null, fallbackTimer = null, boundarySeen = false;

    speedSelect.value = String({rate});
    if (!speedSelect.value) speedSelect.value = '175';
    if (!synth) status.textContent = 'Speech not supported in this browser — PDF view only.';

    function loadVoices() {{
      if (!synth) return;
      const selected = voiceSelect.value;
      const voices = synth.getVoices();
      voiceSelect.innerHTML = '<option value="">System voice</option>';
      voices.forEach((voice, index) => {{
        const option = document.createElement('option');
        option.value = String(index); option.textContent = `${{voice.name}} (${{voice.lang}})`;
        voiceSelect.appendChild(option);
      }});
      if ([...voiceSelect.options].some(o => o.value === selected)) voiceSelect.value = selected;
    }}
    loadVoices(); if (synth) synth.onvoiceschanged = loadVoices;

    function stopSpeech() {{
      if (synth) synth.cancel(); utterance = null; boundarySeen = false;
      if (fallbackTimer) clearInterval(fallbackTimer); fallbackTimer = null;
      setActiveWord(-1); status.textContent = `Page ${{currentPage}} ready`;
      document.getElementById('pause').textContent = 'Pause';
    }}

    function setActiveWord(index) {{
      if (activeWord >= 0) document.getElementById(`word-${{activeWord}}`)?.classList.remove('active');
      activeWord = index;
      const element = document.getElementById(`word-${{index}}`);
      if (element) {{ element.classList.add('active'); element.scrollIntoView({{block:'center', behavior:'smooth'}}); }}
    }}

    function wordAtCharacter(charIndex) {{
      let low = 0, high = wordStarts.length - 1, answer = 0;
      while (low <= high) {{ const mid = (low + high) >> 1; if (wordStarts[mid] <= charIndex) {{ answer = mid; low = mid + 1; }} else high = mid - 1; }}
      return answer;
    }}

    function buildTranscript(text) {{
      transcript.replaceChildren(); wordStarts = [];
      const matches = [...text.matchAll(/\S+/g)];
      matches.forEach((match, index) => {{
        wordStarts.push(match.index);
        const span = document.createElement('span'); span.id = `word-${{index}}`; span.className = 'word'; span.textContent = match[0];
        transcript.appendChild(span); transcript.appendChild(document.createTextNode(' '));
      }});
    }}

    async function renderPage(number) {{
      currentPage = number;
      stopSpeech(); loading.style.display = 'block'; canvas.style.display = 'none'; transcript.textContent = '';
      const page = await pdf.getPage(number);
      const viewport = page.getViewport({{scale: 1.45}});
      canvas.width = viewport.width; canvas.height = viewport.height;
      await page.render({{canvasContext: context, viewport}}).promise;
      const textContent = await page.getTextContent();
      pageText = textContent.items.map(item => item.str).join(' ').replace(/\s+/g, ' ').trim();
      buildTranscript(pageText || 'No selectable text was found on this page.');
      pageNum.textContent = String(number); prev.disabled = number <= 1; next.disabled = number >= pdf.numPages;
      canvas.style.display = 'block'; loading.style.display = 'none'; status.textContent = `Page ${{number}} ready`;
    }}

    function readPage() {{
      if (!pageText || !synth) return;
      stopSpeech(); utterance = new SpeechSynthesisUtterance(pageText);
      utterance.rate = Math.max(.6, Math.min(2, Number(speedSelect.value) / 175));
      const voices = synth.getVoices(); if (voiceSelect.value !== '' && voices[Number(voiceSelect.value)]) utterance.voice = voices[Number(voiceSelect.value)];
      utterance.onstart = () => {{
        status.textContent = `Reading page ${{currentPage}}`;
        setTimeout(() => {{
          if (!boundarySeen && utterance) {{
            let index = Math.max(0, activeWord); const ms = 60000 / Number(speedSelect.value);
            fallbackTimer = setInterval(() => {{ if (++index < wordStarts.length) setActiveWord(index); else clearInterval(fallbackTimer); }}, ms);
          }}
        }}, 900);
      }};
      utterance.onboundary = event => {{ boundarySeen = true; if (fallbackTimer) clearInterval(fallbackTimer); setActiveWord(wordAtCharacter(event.charIndex)); }};
      utterance.onend = () => {{ utterance = null; if (fallbackTimer) clearInterval(fallbackTimer); status.textContent = `Finished page ${{currentPage}}`; }};
      utterance.onerror = event => {{ status.textContent = `Speech error: ${{event.error}}`; }};
      synth.speak(utterance);
    }}

    document.getElementById('read').onclick = readPage;
    document.getElementById('stop').onclick = stopSpeech;
    document.getElementById('pause').onclick = event => {{
      if (!synth || !synth.speaking) return;
      if (synth.paused) {{ synth.resume(); event.currentTarget.textContent = 'Pause'; status.textContent = `Reading page ${{currentPage}}`; }}
      else {{ synth.pause(); event.currentTarget.textContent = 'Resume'; status.textContent = 'Paused'; }}
    }};
    prev.onclick = async () => {{ if (currentPage > 1) await renderPage(currentPage - 1); }};
    next.onclick = async () => {{ if (currentPage < pdf.numPages) await renderPage(currentPage + 1); }};
    window.addEventListener('beforeunload', () => {{ if (synth) synth.cancel(); }});

    try {{
      pdf = await pdfjsLib.getDocument({{data: pdfBytes}}).promise;
      pageTotal.textContent = String(pdf.numPages);
      await renderPage(currentPage);
    }} catch (error) {{
      loading.innerHTML = `<span class="error">Could not load PDF viewer: ${{error.message}}</span>`;
      status.textContent = 'Viewer error'; console.error(error);
    }}
  </script>
</body>
</html>"""
