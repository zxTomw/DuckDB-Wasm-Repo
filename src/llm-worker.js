import {
  InterruptableStoppingCriteria,
  TextStreamer,
  pipeline,
} from '@huggingface/transformers';
import { buildMessages, fitDocumentsToTokenBudget, stripThinking } from './rag.js';

const MODEL_ID = 'Mike0021/MiniCPM5-2B-ONNX';
const MODEL_REVISION = '04a6c49fcba3a65a0351c92644c3a7e9d4343059';
const MAX_INPUT_TOKENS = 3500;

let generator;
let loading;
let activeGeneration;
let generationQueue = Promise.resolve();

function report(type, details = {}) {
  self.postMessage({ type, ...details });
}

async function loadModel() {
  if (generator) return generator;
  if (!loading) {
    loading = pipeline('text-generation', MODEL_ID, {
      device: 'webgpu',
      dtype: 'q4f16',
      revision: MODEL_REVISION,
      progress_callback(progress) {
        report('progress', { progress });
      },
    }).then(value => {
      generator = value;
      report('ready', { model: MODEL_ID, revision: MODEL_REVISION });
      return value;
    }).catch(error => {
      loading = undefined;
      report('error', { operation: 'load', message: error.message });
      throw error;
    });
  }
  return loading;
}

async function countTokens(messages) {
  const prompt = generator.tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
    enable_thinking: false,
  });
  return generator.tokenizer.encode(prompt).length;
}

function generatedText(output, streamed) {
  const generated = output?.[0]?.generated_text;
  if (Array.isArray(generated)) return generated.at(-1)?.content ?? streamed;
  return typeof generated === 'string' ? generated : streamed;
}

async function generate({ requestId, question, documents }) {
  await loadModel();
  const fitted = await fitDocumentsToTokenBudget(
    question,
    documents,
    countTokens,
    MAX_INPUT_TOKENS,
  );
  if (!fitted.length) throw new Error('The retrieved documents do not fit in the model context window.');

  const messages = buildMessages(question, fitted);
  const stoppingCriteria = new InterruptableStoppingCriteria();
  const state = { requestId, stoppingCriteria, cancelled: false };
  activeGeneration = state;
  let streamed = '';
  let visibleLength = 0;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function(text) {
      streamed += text;
      // With thinking disabled, the chat template closes the thinking block
      // in the prompt. skip_prompt omits that closing tag from the stream.
      const visible = stripThinking(streamed);
      if (visible.length > visibleLength) {
        report('answer-delta', { requestId, text: visible.slice(visibleLength) });
        visibleLength = visible.length;
      }
    },
  });

  try {
    const output = await generator(messages, {
      max_new_tokens: 512,
      do_sample: true,
      temperature: 1.0,
      top_p: 0.95,
      top_k: 0,
      repetition_penalty: 1.0,
      streamer,
      stopping_criteria: [stoppingCriteria],
      tokenizer_encode_kwargs: { enable_thinking: false },
    });
    if (state.cancelled) {
      report('cancelled', { requestId });
      return;
    }
    const answer = stripThinking(generatedText(output, streamed));
    if (!answer.trim()) {
      report('error', {
        operation: 'generate',
        requestId,
        message: 'The model produced no final answer. Try searching again or use a shorter question.',
      });
      return;
    }
    report('complete', {
      requestId,
      answer,
      documentIds: fitted.map(document => document.id),
    });
  } catch (error) {
    if (state.cancelled) report('cancelled', { requestId });
    else report('error', { operation: 'generate', requestId, message: error.message });
  } finally {
    if (activeGeneration === state) activeGeneration = undefined;
  }
}

self.onmessage = event => {
  const message = event.data;
  if (message.type === 'load') {
    loadModel().catch(() => {});
    return;
  }
  if (message.type === 'cancel') {
    if (activeGeneration?.requestId === message.requestId) {
      activeGeneration.cancelled = true;
      activeGeneration.stoppingCriteria.interrupt();
    }
    return;
  }
  if (message.type === 'generate') {
    if (activeGeneration) {
      activeGeneration.cancelled = true;
      activeGeneration.stoppingCriteria.interrupt();
    }
    generationQueue = generationQueue
      .catch(() => {})
      .then(() => generate(message));
  }
};
