/**
 * Reader for the Mach Boom weight file (`mach-boom.bin`, magic PHMACH01).
 *
 * Layout: 8-byte magic, u64 little-endian header length, JSON header
 * ({config, tlut, tensors}), then the data section aligned to 64 bytes. Every
 * blob in the header is [offset, length] relative to the data section.
 */

export type Blob2 = [number, number];

export interface TrellisChunk { codes: Blob2; su: Blob2; sv: Blob2; ws: number; m: number; n: number }

export interface TensorEntry {
  type: 'f32' | 'trellis' | 'int8' | 'int4';
  shape: number[];
  data?: Blob2;
  // int8 / int4
  q?: Blob2; s?: Blob2; rows?: number; cols?: number; g?: number;
  // trellis
  K?: number; kind?: 'rows' | 'taps'; convT?: boolean; chunks?: TrellisChunk[];
}

export interface MachConfig {
  lm: {
    hidden: number; layers: number; heads: number; head_dim: number; ffn: number; ln_eps: number;
    rope_theta: number; latent_dim: number; frame_rate: number; flow_layers: number; flow_eps: number;
  };
  gen: {
    temperature: number; eos_threshold: number; frames_after_eos: number;
    frames_after_eos_short_text: number; short_text_max_words: number;
  };
  dec: {
    hidden: number; layers: number; heads: number; head_dim: number; ffn: number; eps: number; window: number;
    rope_theta: number; latent_stride: number; ratios: number[]; kernel: number; last_kernel: number;
    res_kernel: number; sample_rate: number;
  };
}

const MAGIC = 'PHMACH01';

export function f16ToF32(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

export class MachPack {
  u8: Uint8Array<ArrayBuffer>;
  base: number;
  readonly cfg: MachConfig;
  T: Record<string, TensorEntry>;
  tlutBlobs: Record<string, Blob2>;

  constructor(buf: ArrayBuffer) {
    this.u8 = new Uint8Array(buf);
    const magic = new TextDecoder().decode(this.u8.subarray(0, 8));
    if (magic !== MAGIC) throw new Error('Not a Mach Boom weight file');
    const dv = new DataView(buf);
    const hl = Number(dv.getBigUint64(8, true));
    const H = JSON.parse(new TextDecoder().decode(this.u8.subarray(16, 16 + hl)));
    this.base = Math.ceil((16 + hl) / 64) * 64;
    this.cfg = H.config;
    this.T = H.tensors;
    this.tlutBlobs = H.tlut;
  }

  has(name: string): boolean { return name in this.T; }

  /**
   * Keep only `names` (their blobs copied into a compact buffer) and drop the
   * rest, so the original weight-file ArrayBuffer is no longer referenced.
   */
  retain(names: string[]) {
    const T: Record<string, TensorEntry> = {};
    const parts: Uint8Array<ArrayBuffer>[] = [];
    let size = 0;
    const move = (b: Blob2): Blob2 => {
      const bytes = this.blob(b), off = size;
      parts.push(bytes);
      size += Math.ceil(bytes.length / 8) * 8;
      return [off, bytes.length];
    };
    for (const n of names) {
      const e = { ...this.entry(n) };
      if (e.data) e.data = move(e.data);
      if (e.q) e.q = move(e.q);
      if (e.s) e.s = move(e.s);
      if (e.chunks) throw new Error(`${n}: cannot retain trellis tensors`);
      T[n] = e;
    }
    const u8 = new Uint8Array(size);
    let o = 0;
    for (const p of parts) { u8.set(p, o); o += Math.ceil(p.length / 8) * 8; }
    this.u8 = u8;
    this.base = 0;
    this.T = T;
    this.tlutBlobs = {};
  }

  entry(name: string): TensorEntry {
    const e = this.T[name];
    if (!e) throw new Error(`Mach Boom: missing tensor ${name}`);
    return e;
  }

  /** Raw bytes of a blob (a view into the file; copy before keeping). */
  blob([o, n]: Blob2): Uint8Array<ArrayBuffer> {
    return this.u8.subarray(this.base + o, this.base + o + n);
  }

  /** Copy of a blob as Float32Array (handles any alignment). */
  blobF32(b: Blob2): Float32Array<ArrayBuffer> {
    const bytes = this.blob(b);
    const out = new Float32Array(bytes.length / 4);
    new Uint8Array(out.buffer).set(bytes);
    return out;
  }

  f32(name: string): Float32Array<ArrayBuffer> {
    const e = this.entry(name);
    if (e.type !== 'f32' || !e.data) throw new Error(`${name} is not f32`);
    return this.blobF32(e.data);
  }

  private f16s(b: Blob2): Float32Array<ArrayBuffer> {
    const bytes = this.blob(b);
    const out = new Float32Array(bytes.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = f16ToF32(bytes[2 * i]! | (bytes[2 * i + 1]! << 8));
    return out;
  }

  /** Dequantized int8 tensor as a row-major [rows][cols] matrix. */
  small(name: string): Float32Array<ArrayBuffer> {
    const e = this.entry(name);
    if (e.type !== 'int8') throw new Error(`${name} is not int8`);
    const rows = e.rows!, cols = e.cols!, g = e.g!;
    const q = this.blob(e.q!);
    const s = this.f16s(e.s!);
    const gpr = cols / g;
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = (q[r * cols + c]! << 24) >> 24;
        out[r * cols + c] = v * s[r * gpr + ((c / g) | 0)]!;
      }
    return out;
  }

  /** One dequantized row of the int4 token embedding. */
  embedRow(i: number, out = new Float32Array(this.entry('lm:model.embed_tokens').cols!)): Float32Array<ArrayBuffer> {
    const e = this.entry('lm:model.embed_tokens');
    const n = e.cols!, g = e.g!;
    const q = this.blob(e.q!);
    const sb = this.blob(e.s!);
    const qo = (i * n) / 2;
    for (let j = 0; j < n; j++) {
      const b = q[qo + (j >> 1)]!;
      let v = j & 1 ? b >> 4 : b & 0xf;
      if (v > 7) v -= 16;
      const si = (i * n) / g + ((j / g) | 0);
      out[j] = v * f16ToF32(sb[2 * si]! | (sb[2 * si + 1]! << 8));
    }
    return out;
  }

  /** The 1024x2 decode table for a trellis bit width (tlut10 in packfmt.py). */
  tlut10(K: number): Float32Array<ArrayBuffer> {
    const tl = this.blobF32(this.tlutBlobs[String(K)]!);
    const t = new Float32Array(2048);
    for (let e = 0; e < 1024; e++) {
      const x = tl[2 * (e & 511)]!, y = tl[2 * (e & 511) + 1]!;
      t[2 * e] = e >> 9 ? -x : x;
      t[2 * e + 1] = y;
    }
    return t;
  }
}

/** speaker_wavs [1, 512, P] from a voice .safetensors file, returned as [P][512]. */
export function parseVoice(buf: ArrayBuffer, D = 512): { rows: number; data: Float32Array<ArrayBuffer> } {
  const dv = new DataView(buf);
  const n = Number(dv.getBigUint64(0, true));
  const H = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const e = H.speaker_wavs;
  if (!e) throw new Error('voice file has no speaker_wavs');
  const [o0, o1] = e.data_offsets as [number, number];
  const bytes = new Uint8Array(buf, 8 + n + o0, o1 - o0);
  let src: Float32Array<ArrayBuffer>;
  if (e.dtype === 'F32') {
    src = new Float32Array(bytes.length / 4);
    new Uint8Array(src.buffer).set(bytes);
  } else if (e.dtype === 'F16') {
    src = new Float32Array(bytes.length / 2);
    for (let i = 0; i < src.length; i++) src[i] = f16ToF32(bytes[2 * i]! | (bytes[2 * i + 1]! << 8));
  } else throw new Error(`unsupported voice dtype ${e.dtype}`);
  const shape = e.shape as number[];
  const C = shape[1]!, P = shape[2]!;
  if (C !== D) throw new Error(`voice width ${C} != ${D}`);
  const out = new Float32Array(P * C);
  for (let c = 0; c < C; c++) for (let p = 0; p < P; p++) out[p * C + c] = src[c * P + p]!;
  return { rows: P, data: out };
}
