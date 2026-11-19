export { MachBoomEngine, type SynthOptions, type SynthResult } from './engine';
export { normalize } from './frontend';
export { UnigramTokenizer } from './tokenizer';

export { MACH_BOOM_VOICES, VOICE_IDS, DEFAULT_VOICE, VOICE_BASE_URL } from './voices';

/** Split long text into sentence-sized chunks (at most `limit` chars). */
export function splitText(text: string, limit = 280): string[] {
  const normalized = text.replace(/\s+/g, ' ').trim();
  // Don't break after titles/abbreviations ("Dr. Smith", "Jan. 5"): the
  // frontend can only expand them with the following word in the same chunk.
  const sentences = normalized
    .split(/(?<=[.!?])(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Gen|Col|Capt|Lt|Sgt|Rev|Hon|vs|No|Fig|Vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|e\.g|i\.e)\.)\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  for (const sentence of sentences.length ? sentences : [normalized]) {
    let rest = sentence;
    while (rest.length > limit) {
      const search = rest.slice(0, limit + 1);
      let punctuation = -1;
      for (const mark of [',', ';', ':']) punctuation = Math.max(punctuation, search.lastIndexOf(mark));
      let splitAt = punctuation >= limit / 2 ? punctuation + 1 : rest.lastIndexOf(' ', limit + 1);
      if (splitAt < limit / 2) splitAt = limit;
      chunks.push(rest.slice(0, splitAt).trim());
      rest = rest.slice(splitAt).trim();
    }
    if (rest) chunks.push(rest);
  }
  return chunks;
}

/** Inter-chunk pause lengths by trailing punctuation of previous chunk. */
export function boundaryPauseSeconds(chunk: string): number {
  const ending = chunk.trim().slice(-1);
  switch (ending) {
    case '?': return 0.28;
    case '!': return 0.24;
    case '.': return 0.22;
    case ';': return 0.16;
    case ':': return 0.13;
    case ',': return 0.09;
    default: return 0.08;
  }
}

/** 5 ms linear edge fade to avoid clicks between chunks. */
export function edgeFade(waveform: Float32Array<ArrayBuffer>, sampleRate: number, milliseconds = 5): Float32Array<ArrayBuffer> {
  const frames = Math.min(Math.round((sampleRate * milliseconds) / 1000), Math.floor(waveform.length / 2));
  if (frames <= 0) return waveform;
  const out = waveform.slice();
  for (let j = 0; j < frames; j++) {
    const ramp = j / (frames - 1 || 1);
    out[j]! *= ramp;
    out[out.length - 1 - j]! *= ramp;
  }
  return out;
}

/**
 * Pitch-preserving time stretch (WSOLA). `speed` > 1 shortens the audio.
 * Mach Boom has no duration control, so the reader's speed setting is applied here.
 */
export function timeStretch(x: Float32Array<ArrayBuffer>, speed: number, sampleRate: number): Float32Array<ArrayBuffer> {
  if (!(speed > 0) || Math.abs(speed - 1) < 0.01) return x;
  const N = Math.round(sampleRate * 0.03) & ~1; // 30 ms frames
  const Hs = N / 2; // 50% overlap at the output
  const tol = Math.round(sampleRate * 0.008); // ±8 ms search
  if (x.length < 2 * N) return x;
  const win = new Float32Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  const outLen = Math.round(x.length / speed);
  const y = new Float32Array(outLen + N);
  const wsum = new Float32Array(outLen + N);
  const last = x.length - N;
  const L = Hs; // correlation length
  let prev = 0;
  for (let outPos = 0; outPos < outLen; outPos += Hs) {
    let pos: number;
    if (outPos === 0) pos = 0;
    else {
      const ref = Math.min(prev + Hs, last); // natural continuation of the previous frame
      const target = Math.round(outPos * speed);
      const lo = Math.max(0, target - tol), hi = Math.min(last, target + tol);
      pos = Math.min(Math.max(target, 0), last);
      let best = -Infinity;
      for (let p = lo; p <= hi; p++) {
        let s = 0;
        for (let i = 0; i < L; i += 2) s += x[p + i]! * x[ref + i]!;
        if (s > best) { best = s; pos = p; }
      }
    }
    for (let i = 0; i < N; i++) {
      y[outPos + i]! += x[pos + i]! * win[i]!;
      wsum[outPos + i]! += win[i]!;
    }
    prev = pos;
  }
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) out[i] = wsum[i]! > 1e-3 ? y[i]! / wsum[i]! : y[i]!;
  return out;
}
