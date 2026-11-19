// tts.worker.ts — Mach Boom TTS (SyzygyResearch/Mach-Boom) on WebGPU, 24 kHz
import {
  MachBoomEngine, VOICE_IDS, DEFAULT_VOICE, VOICE_BASE_URL, splitText, boundaryPauseSeconds, edgeFade, timeStretch,
} from './mach-boom-tts/index.ts';

const HF_BASE = 'https://huggingface.co/SyzygyResearch/Mach-Boom/resolve/main';
const MODEL_DB = 'mach-boom-models-v1';
const WEIGHTS = 'mach-boom.bin';
const TOKENIZER = 'tokenizer.json';

// IDB helpers — IndexedDB stores raw ArrayBuffer so there are no CORS or
// redirect restrictions (unlike Cache Storage, which rejects cross-origin
// redirected responses from dedicated workers).
function openModelDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(MODEL_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore('models');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function idbGet(db: IDBDatabase, key: string): Promise<ArrayBuffer | undefined> {
  return new Promise((resolve, reject) => {
    const req = db.transaction('models', 'readonly').objectStore('models').get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function idbPut(db: IDBDatabase, key: string, buf: ArrayBuffer): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('models', 'readwrite');
    tx.objectStore('models').put(buf, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function fetchCached(db: IDBDatabase, url: string, key: string): Promise<ArrayBuffer> {
  const cached = await idbGet(db, key);
  if (cached) return cached;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${key}: ${res.status}`);
  const buf = await res.arrayBuffer();
  await idbPut(db, key, buf);
  return buf;
}
const fetchModelFile = (db: IDBDatabase, path: string) => fetchCached(db, `${HF_BASE}/${path}`, path);
// Voices come from gradium/phonon-voices (Mach Boom's own repo only ships four of them).
const fetchVoice = (db: IDBDatabase, id: string) => fetchCached(db, `${VOICE_BASE_URL}/${id}.safetensors`, `voices/${id}.safetensors`);

let engine: MachBoomEngine | null = null;
let db: IDBDatabase | null = null;
let generateChain: Promise<void> = Promise.resolve();

async function ensureVoice(name: string | undefined): Promise<string> {
  const voice = name && VOICE_IDS[name] ? name : DEFAULT_VOICE;
  if (!engine!.hasVoice(voice)) {
    db ??= await openModelDB();
    engine!.addVoice(voice, await fetchVoice(db, VOICE_IDS[voice]!));
  }
  return voice;
}

async function runGenerate(text: string, lineIndex: number | undefined, voiceName: string | undefined, speed: number, notify: boolean) {
  if (!engine) throw new Error('TTS not initialized');
  const voice = await ensureVoice(voiceName);
  const sr = engine.sampleRate;
  const chunks = splitText(text);
  const pieces: Float32Array[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const { audio } = await engine.synthesize(chunks[i]!, voice);
    if (!audio.length) continue;
    if (pieces.length) pieces.push(new Float32Array(Math.round(sr * boundaryPauseSeconds(chunks[i - 1]!))));
    pieces.push(edgeFade(audio, sr));
  }
  if (!notify) return;
  if (!pieces.length) throw new Error('The text frontend produced no speakable tokens.');
  let total = 0;
  for (const p of pieces) total += p.length;
  let audio = new Float32Array(total);
  let off = 0;
  for (const p of pieces) { audio.set(p, off); off += p.length; }
  audio = timeStretch(audio, speed, sr);
  self.postMessage({ status: 'complete', audio, text, lineIndex }, [audio.buffer]);
}

self.addEventListener('message', (e: MessageEvent<any>) => {
  const { type } = e.data;
  generateChain = generateChain.then(async () => {
    try {
      if (type === 'init') {
        if (engine) return;

        self.postMessage({ status: 'loading', message: 'Initializing WebGPU...' });
        const device = await MachBoomEngine.requestDevice();
        device.lost.then((info) => {
          console.warn('[MachBoom Worker] GPU device lost', info.message);
          engine = null;
        });

        db = await openModelDB();
        const alreadyCached = !!(await idbGet(db, WEIGHTS));
        self.postMessage({ status: 'loading', message: alreadyCached ? 'Loading Model...' : 'Downloading Mach Boom (18 MB)...' });
        const [weights, tokenizer] = await Promise.all([
          fetchModelFile(db, WEIGHTS),
          fetchModelFile(db, TOKENIZER).then((b) => JSON.parse(new TextDecoder().decode(b))),
          fetchVoice(db, VOICE_IDS[DEFAULT_VOICE]!),
        ]);
        self.postMessage({ status: 'models_cached' });

        engine = await MachBoomEngine.create(weights, tokenizer, device);

        // Compile pipelines and build the voice prefix before real lines arrive.
        self.postMessage({ status: 'loading', message: 'Warming up…' });
        await runGenerate('Warm up.', -1, DEFAULT_VOICE, 1.0, false);

        console.log('[MachBoom Worker] Ready');
        self.postMessage({ status: 'ready', device: 'webgpu', dtype: 'fp16' });
      } else if (type === 'generate') {
        const { text, lineIndex, voice, speed = 1.0 } = e.data;
        if (!engine) throw new Error('TTS not initialized (GPU device was lost — reload the page)');
        await runGenerate(text, lineIndex, voice, speed, true);
      }
    } catch (err: any) {
      console.error('[MachBoom Worker]', err);
      if (type === 'init') engine = null;
      const isFetchError = err instanceof TypeError && err.message.toLowerCase().includes('fetch');
      self.postMessage({
        status: 'error',
        error: err?.message || String(err) || 'Unknown error',
        fetchFailed: isFetchError,
      });
    }
  });
});
