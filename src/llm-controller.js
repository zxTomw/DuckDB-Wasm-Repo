export async function detectWebGPU() {
  if (!globalThis.navigator?.gpu) {
    return { supported: false, reason: 'WebGPU is unavailable in this browser.' };
  }
  const adapter = await globalThis.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { supported: false, reason: 'No WebGPU adapter is available.' };
  if (!adapter.features.has('shader-f16')) {
    return { supported: false, reason: 'This GPU does not expose the shader-f16 feature required by the model.' };
  }
  return { supported: true };
}

function defaultWorkerFactory() {
  return new Worker(new URL('./llm-worker.js', import.meta.url), { type: 'module' });
}

function progressPercent(progress) {
  if (Number.isFinite(progress?.progress)) return Math.min(100, Math.max(0, progress.progress));
  if (Number.isFinite(progress?.loaded) && Number.isFinite(progress?.total) && progress.total > 0) {
    return Math.min(100, Math.max(0, progress.loaded / progress.total * 100));
  }
  return null;
}

export function renderAnswer(container, text, allowedIds = []) {
  const value = String(text ?? '');
  const allowed = new Set(allowedIds.map(String));
  if (!container.ownerDocument || typeof container.replaceChildren !== 'function') {
    container.textContent = value;
    return;
  }
  const nodes = [];
  let position = 0;
  for (const match of value.matchAll(/\[([A-Za-z0-9_.:-]+)\]/g)) {
    if (match.index > position) nodes.push(container.ownerDocument.createTextNode(value.slice(position, match.index)));
    const id = match[1];
    if (allowed.has(id)) {
      const link = container.ownerDocument.createElement('a');
      link.href = `#fts-result-${encodeURIComponent(id)}`;
      link.textContent = match[0];
      link.title = `Jump to retrieved document ${id}`;
      nodes.push(link);
    } else {
      nodes.push(container.ownerDocument.createTextNode(match[0]));
    }
    position = match.index + match[0].length;
  }
  if (position < value.length) nodes.push(container.ownerDocument.createTextNode(value.slice(position)));
  container.replaceChildren(...nodes);
}

export class LLMController {
  constructor({
    elements,
    workerFactory = defaultWorkerFactory,
    detectWebGPU: capabilityDetector = detectWebGPU,
  }) {
    this.elements = elements;
    this.workerFactory = workerFactory;
    this.capabilityDetector = capabilityDetector;
    this.worker = null;
    this.capability = null;
    this.state = 'checking';
    this.activeRequestId = null;
    this.requestNumber = 0;
    this.answerText = '';
    this.allowedIds = [];
  }

  get ready() {
    return this.state === 'ready' || this.state === 'generating';
  }

  async initializeCapability() {
    this.state = 'checking';
    this.elements.loadButton.disabled = true;
    this.elements.stopButton.disabled = true;
    this.elements.status.textContent = 'Checking WebGPU support…';
    try {
      this.capability = await this.capabilityDetector();
    } catch (error) {
      this.capability = { supported: false, reason: error.message };
    }
    if (!this.capability.supported) {
      this.state = 'unsupported';
      this.elements.status.textContent = `${this.capability.reason} BM25 search remains available.`;
      this.elements.loadButton.disabled = true;
      return this.capability;
    }
    this.state = 'idle';
    this.elements.status.textContent = 'WebGPU is ready. Load the local model when you want grounded answers.';
    this.elements.loadButton.disabled = false;
    return this.capability;
  }

  ensureWorker() {
    if (this.worker) return this.worker;
    this.worker = this.workerFactory();
    this.worker.onmessage = event => this.handleMessage(event.data);
    this.worker.onerror = event => {
      event.preventDefault?.();
      this.handleMessage({
        type: 'error',
        operation: this.state === 'loading' ? 'load' : 'generate',
        requestId: this.activeRequestId,
        message: event.message || 'The LLM worker stopped unexpectedly.',
      });
    };
    return this.worker;
  }

  async load() {
    if (!this.capability) await this.initializeCapability();
    if (!this.capability?.supported || this.state === 'loading' || this.ready) return false;
    this.state = 'loading';
    this.elements.loadButton.disabled = true;
    this.elements.stopButton.disabled = true;
    this.elements.progress.hidden = false;
    this.elements.progress.removeAttribute?.('value');
    this.elements.status.textContent = 'Starting the local model download…';
    this.ensureWorker().postMessage({ type: 'load' });
    return true;
  }

  generate(question, documents) {
    if (!this.ready || this.state === 'loading') {
      const message = this.state === 'unsupported'
        ? 'BM25 results are ready. Local answer generation is unavailable on this device.'
        : 'BM25 results are ready. Load the local LLM to generate a grounded answer.';
      renderAnswer(this.elements.answer, message);
      return false;
    }
    if (this.activeRequestId) this.cancel(true);
    const requestId = `rag-${++this.requestNumber}`;
    this.activeRequestId = requestId;
    this.state = 'generating';
    this.answerText = '';
    this.allowedIds = documents.map(document => String(document.id));
    renderAnswer(this.elements.answer, '');
    this.elements.status.textContent = 'Generating a grounded answer locally…';
    this.elements.stopButton.disabled = false;
    this.ensureWorker().postMessage({
      type: 'generate',
      requestId,
      question,
      documents: documents.map(document => ({
        id: String(document.id),
        title: String(document.title ?? ''),
        text: String(document.text ?? ''),
      })),
    });
    return true;
  }

  showRetrievalMessage(message) {
    if (this.activeRequestId) this.cancel(true);
    renderAnswer(this.elements.answer, message);
  }

  cancel(quiet = false) {
    if (!this.activeRequestId || !this.worker) return false;
    const requestId = this.activeRequestId;
    this.worker.postMessage({ type: 'cancel', requestId });
    this.activeRequestId = null;
    this.state = 'ready';
    this.elements.stopButton.disabled = true;
    if (!quiet) this.elements.status.textContent = 'Generation stopped. BM25 results remain available.';
    return true;
  }

  handleMessage(message) {
    if (message.type === 'progress' && this.state === 'loading') {
      const percent = progressPercent(message.progress);
      const file = message.progress?.file ? ` ${message.progress.file}` : '';
      if (percent === null) {
        this.elements.progress.removeAttribute?.('value');
        this.elements.status.textContent = `Loading model${file}…`;
      } else {
        this.elements.progress.max = 100;
        this.elements.progress.value = percent;
        this.elements.status.textContent = `Loading model${file}… ${percent.toFixed(0)}%`;
      }
      return;
    }
    if (message.type === 'ready') {
      this.state = 'ready';
      this.elements.progress.hidden = true;
      this.elements.loadButton.disabled = true;
      this.elements.stopButton.disabled = true;
      this.elements.status.textContent = 'Local MiniCPM5-2B model ready. Searches will now generate grounded answers.';
      return;
    }
    if (message.requestId && message.requestId !== this.activeRequestId) return;
    if (message.type === 'answer-delta') {
      this.answerText += message.text;
      renderAnswer(this.elements.answer, this.answerText, this.allowedIds);
      return;
    }
    if (message.type === 'complete') {
      this.answerText = message.answer || 'The model produced no final answer. Try searching again.';
      renderAnswer(this.elements.answer, this.answerText, this.allowedIds);
      this.activeRequestId = null;
      this.state = 'ready';
      this.elements.stopButton.disabled = true;
      this.elements.status.textContent = `Grounded answer generated from ${message.documentIds?.length ?? 0} retrieved documents.`;
      return;
    }
    if (message.type === 'cancelled') {
      this.activeRequestId = null;
      this.state = 'ready';
      this.elements.stopButton.disabled = true;
      this.elements.status.textContent = 'Generation stopped. BM25 results remain available.';
      return;
    }
    if (message.type === 'error') {
      if (message.operation === 'load') {
        this.state = 'error';
        this.elements.progress.hidden = true;
        this.elements.loadButton.disabled = false;
        this.elements.status.textContent = `Model load failed: ${message.message}. You can retry.`;
      } else {
        this.activeRequestId = null;
        this.state = 'ready';
        this.elements.stopButton.disabled = true;
        this.elements.status.textContent = `Answer generation failed: ${message.message}. BM25 results remain available.`;
      }
    }
  }

  dispose() {
    if (this.activeRequestId) this.cancel(true);
    this.worker?.terminate();
    this.worker = null;
  }
}

export function setupLLM() {
  const elements = {
    loadButton: document.querySelector('#llm-load'),
    stopButton: document.querySelector('#llm-stop'),
    status: document.querySelector('#llm-status'),
    progress: document.querySelector('#llm-progress'),
    answer: document.querySelector('#llm-answer'),
  };
  const controller = new LLMController({ elements });
  elements.loadButton.onclick = () => controller.load();
  elements.stopButton.onclick = () => controller.cancel();
  controller.initializeCapability();
  return controller;
}
