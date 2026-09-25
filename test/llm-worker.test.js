import { beforeEach, describe, expect, it, vi } from 'vitest';

const { generate, pipeline } = vi.hoisted(() => ({
  generate: vi.fn(),
  pipeline: vi.fn(),
}));

vi.mock('@huggingface/transformers', () => ({
  pipeline,
  InterruptableStoppingCriteria: class { interrupt() {} },
  TextStreamer: class {
    constructor(_tokenizer, options) {
      this.callback = options.callback_function;
    }
  },
}));

describe('MiniCPM worker', () => {
  beforeEach(() => {
    vi.resetModules();
    generate.mockReset();
    pipeline.mockReset();
    globalThis.self = { postMessage: vi.fn() };
    pipeline.mockResolvedValue(Object.assign(generate, {
      tokenizer: {
        apply_chat_template: vi.fn(() => 'prompt'),
        encode: vi.fn(() => [1, 2]),
      },
    }));
  });

  it('disables thinking in both prompt sizing and generation, then streams the direct answer', async () => {
    generate.mockImplementation(async (_messages, options) => {
      options.streamer.callback('Answer [MED-14]');
      return [{ generated_text: [{ content: 'Answer [MED-14]' }] }];
    });
    await import('../src/llm-worker.js');
    self.onmessage({ data: {
      type: 'generate', requestId: 'one', question: 'What?',
      documents: [{ id: 'MED-14', title: 'Title', text: 'Evidence' }],
    } });
    await vi.waitFor(() => expect(self.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'complete', answer: 'Answer [MED-14]' }),
    ));

    const tokenizer = (await pipeline.mock.results[0].value).tokenizer;
    expect(tokenizer.apply_chat_template).toHaveBeenCalledWith(
      expect.any(Array), expect.objectContaining({ enable_thinking: false }),
    );
    expect(generate.mock.calls[0][1].tokenizer_encode_kwargs).toEqual({ enable_thinking: false });
    expect(self.postMessage).toHaveBeenCalledWith({
      type: 'answer-delta', requestId: 'one', text: 'Answer [MED-14]',
    });
  });

  it('reports an empty final answer as a generation error', async () => {
    generate.mockResolvedValue([{ generated_text: [{ content: '<think>unfinished reasoning' }] }]);
    await import('../src/llm-worker.js');
    self.onmessage({ data: {
      type: 'generate', requestId: 'two', question: 'What?',
      documents: [{ id: 'MED-14', title: 'Title', text: 'Evidence' }],
    } });
    await vi.waitFor(() => expect(self.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'error', operation: 'generate', requestId: 'two' }),
    ));
    expect(self.postMessage).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'complete' }));
  });
});
