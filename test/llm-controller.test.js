import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LLMController, renderAnswer } from '../src/llm-controller.js';

class FakeEventTarget {
  constructor() {
    this.disabled = false;
    this.hidden = false;
    this.textContent = '';
    this.value = 0;
    this.max = 100;
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter(value => value !== listener));
  }

  dispatch(type) {
    for (const listener of this.listeners.get(type) ?? []) listener({ type, target: this });
  }
}

class MockWorker {
  constructor() {
    this.messages = [];
    this.listeners = new Map();
    this.onmessage = null;
    this.terminated = false;
  }

  postMessage(message) {
    this.messages.push(message);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter(value => value !== listener));
  }

  emit(data) {
    const event = { data };
    this.onmessage?.(event);
    for (const listener of this.listeners.get('message') ?? []) listener(event);
  }

  terminate() {
    this.terminated = true;
  }
}

function createHarness(capability = { supported: true }) {
  const worker = new MockWorker();
  const elements = {
    loadButton: new FakeEventTarget(),
    stopButton: new FakeEventTarget(),
    status: new FakeEventTarget(),
    progress: new FakeEventTarget(),
    answer: new FakeEventTarget(),
  };
  const workerFactory = vi.fn(() => worker);
  const detectWebGPU = vi.fn(async () => capability);
  const controller = new LLMController({ workerFactory, elements, detectWebGPU });
  return { controller, detectWebGPU, elements, worker, workerFactory };
}

describe('LLMController', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('keeps BM25-only mode usable when WebGPU is unsupported', async () => {
    const harness = createHarness({ supported: false, reason: 'WebGPU is unavailable.' });

    await harness.controller.initializeCapability();
    await harness.controller.load();

    expect(harness.workerFactory).not.toHaveBeenCalled();
    expect(harness.elements.loadButton.disabled).toBe(true);
    expect(harness.elements.status.textContent).toMatch(/WebGPU|unavailable|unsupported/i);
  });

  it('loads only after an explicit request and reflects progress and ready messages', async () => {
    const { controller, elements, worker, workerFactory } = createHarness();

    await controller.initializeCapability();
    expect(workerFactory).not.toHaveBeenCalled();

    await controller.load();
    expect(worker.messages).toContainEqual({ type: 'load' });

    worker.emit({ type: 'progress', progress: { progress: 50, total: 100, file: 'model.onnx' } });
    expect(elements.status.textContent).toMatch(/50|model\.onnx|download/i);

    worker.emit({ type: 'ready' });
    expect(elements.status.textContent).toMatch(/ready/i);
    expect(elements.loadButton.disabled).toBe(true);
  });

  it('streams only the active request and ignores stale worker messages', async () => {
    const { controller, elements, worker } = createHarness();
    await controller.initializeCapability();
    await controller.load();
    worker.emit({ type: 'ready' });

    controller.generate('first', [{ id: 'MED-14', title: 'One', text: 'First' }]);
    const first = worker.messages.find(message => message.type === 'generate');
    worker.emit({ type: 'answer-delta', requestId: first.requestId, text: 'Current ' });
    expect(elements.answer.textContent).toBe('Current ');

    controller.generate('second', [{ id: 'MED-2', title: 'Two', text: 'Second' }]);
    const generations = worker.messages.filter(message => message.type === 'generate');
    const second = generations.at(-1);
    expect(second.requestId).not.toBe(first.requestId);
    expect(worker.messages).toContainEqual({ type: 'cancel', requestId: first.requestId });

    worker.emit({ type: 'answer-delta', requestId: first.requestId, text: 'stale' });
    worker.emit({ type: 'complete', requestId: first.requestId, answer: 'stale complete' });
    expect(elements.answer.textContent).not.toContain('stale');

    worker.emit({ type: 'answer-delta', requestId: second.requestId, text: 'Fresh' });
    worker.emit({ type: 'complete', requestId: second.requestId, answer: 'Fresh [MED-2]' });
    expect(elements.answer.textContent).toBe('Fresh [MED-2]');
  });

  it('cancels explicitly and suppresses late output from the cancelled request', async () => {
    const { controller, elements, worker } = createHarness();
    await controller.initializeCapability();
    await controller.load();
    worker.emit({ type: 'ready' });
    controller.generate('question', [{ id: 'MED-14', title: 'One', text: 'Evidence' }]);
    const request = worker.messages.find(message => message.type === 'generate');

    controller.cancel();
    expect(worker.messages).toContainEqual({ type: 'cancel', requestId: request.requestId });

    worker.emit({ type: 'answer-delta', requestId: request.requestId, text: 'too late' });
    worker.emit({ type: 'complete', requestId: request.requestId, answer: 'too late' });
    expect(elements.answer.textContent).not.toContain('too late');
    expect(elements.stopButton.disabled).toBe(true);
  });

  it('does not confuse empty model output with insufficient evidence', async () => {
    const { controller, elements, worker } = createHarness();
    await controller.initializeCapability();
    await controller.load();
    worker.emit({ type: 'ready' });
    controller.generate('unanswerable', [{ id: 'MED-14', title: 'One', text: 'Evidence' }]);
    const request = worker.messages.find(message => message.type === 'generate');

    worker.emit({ type: 'complete', requestId: request.requestId, answer: '', documentIds: ['MED-14'] });

    expect(elements.answer.textContent).toMatch(/produced no final answer/i);
  });

  it('allows retry after a model-load error and terminates its worker on disposal', async () => {
    const { controller, elements, worker } = createHarness();
    await controller.initializeCapability();
    await controller.load();
    worker.emit({ type: 'error', operation: 'load', message: 'download failed' });

    expect(elements.status.textContent).toMatch(/download failed/i);
    expect(elements.loadButton.disabled).toBe(false);

    await controller.load();
    expect(worker.messages.filter(message => message.type === 'load')).toHaveLength(2);
    controller.dispose();
    expect(worker.terminated).toBe(true);
  });
});

describe('renderAnswer', () => {
  it('links only citations from the retrieved result set and never parses model HTML', () => {
    const document = {
      createTextNode: text => ({ nodeType: 'text', textContent: text }),
      createElement: tagName => ({ tagName, textContent: '', href: '', title: '' }),
    };
    const container = {
      ownerDocument: document,
      children: [],
      replaceChildren(...children) { this.children = children; },
    };

    renderAnswer(container, '<b>Claim</b> [MED-14] and [MED-404].', ['MED-14']);

    const link = container.children.find(node => node.tagName === 'a');
    expect(link.textContent).toBe('[MED-14]');
    expect(link.href).toBe('#fts-result-MED-14');
    expect(container.children.filter(node => node.tagName)).toHaveLength(1);
    expect(container.children.map(node => node.textContent).join('')).toBe(
      '<b>Claim</b> [MED-14] and [MED-404].',
    );
  });
});
