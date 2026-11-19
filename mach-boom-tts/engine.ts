/// <reference types="@webgpu/types" />
/**
 * Mach Boom text-to-speech on WebGPU.
 *
 * All weights are expanded to dense packed-f16 on the GPU at load time (the
 * trellis codes are decoded by two compute passes per chunk into an f32 stage,
 * then packed), then the language model, flow head and Mimi-style decoder run
 * as fixed dispatch lists over one f32 activation buffer plus one f16 KV-cache
 * buffer per graph.
 */
import { MachPack, parseVoice } from './pack';
import { UnigramTokenizer } from './tokenizer';
import { normalize } from './frontend';
import { BINDING_NAMES, KERNELS, kernelSource, type KernelDef } from './shaders';

const U = () => GPUBufferUsage;
const KMAP = Object.fromEntries(KERNELS.map((k) => [k.name, k]));
const SEA = 'dec:decoder.decoder_seanet.layers.';

type Params = Record<string, number>;
interface Dispatch { pipe: GPUComputePipeline; bg: GPUBindGroup; wg: [number, number, number] }

class Program {
  readonly list: Dispatch[] = [];
  readonly uniforms: GPUBuffer[] = [];
  encode(pass: GPUComputePassEncoder) {
    for (const d of this.list) {
      pass.setPipeline(d.pipe);
      pass.setBindGroup(0, d.bg);
      pass.dispatchWorkgroups(d.wg[0], d.wg[1], d.wg[2]);
    }
  }
  destroy() { for (const u of this.uniforms) u.destroy(); this.uniforms.length = 0; this.list.length = 0; }
}

/** Packs small f32 vectors (biases, norms, scales) into one read-only buffer. */
class ConstPool {
  private parts: Float32Array<ArrayBuffer>[] = [];
  private size = 0;
  add(v: ArrayLike<number>): number {
    const off = this.size;
    const a = Float32Array.from(v as ArrayLike<number>);
    this.parts.push(a);
    this.size += a.length;
    const pad = (4 - (this.size % 4)) % 4; // keep vec4-friendly alignment
    if (pad) { this.parts.push(new Float32Array(pad)); this.size += pad; }
    return off;
  }
  build(device: GPUDevice): GPUBuffer {
    const all = new Float32Array(Math.max(4, this.size));
    let o = 0;
    for (const p of this.parts) { all.set(p, o); o += p.length; }
    const buf = device.createBuffer({ size: all.byteLength, usage: U().STORAGE | U().COPY_DST });
    device.queue.writeBuffer(buf, 0, all);
    return buf;
  }
}

/** Packed [cos(32) | sin(32)] rows for positions [0, n). */
function ropeTable(theta: number, n: number, hd = 64): Float32Array<ArrayBuffer> {
  const half = hd / 2;
  const inv = new Float32Array(half);
  for (let i = 0; i < half; i++) inv[i] = 1 / theta ** ((2 * i) / hd);
  const t = new Float32Array(n * hd);
  for (let p = 0; p < n; p++) {
    for (let i = 0; i < half; i++) {
      const a = Math.fround(Math.fround(p) * inv[i]!);
      t[p * hd + i] = Math.cos(a);
      t[p * hd + half + i] = Math.sin(a);
    }
  }
  return t;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(n: number, seed: number, sd: number): Float32Array<ArrayBuffer> {
  const r = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(r(), 1e-12), u2 = r();
    const m = Math.sqrt(-2 * Math.log(u1)) * sd;
    out[i] = m * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

const f32b = new Float32Array(1), u32b = new Uint32Array(f32b.buffer);
/** IEEE f32 -> f16 bits, round to nearest even (matches WGSL pack2x16float on finite values). */
function f16Bits(v: number): number {
  f32b[0] = v;
  const x = u32b[0]!, sign = (x >>> 16) & 0x8000, E = (x >>> 23) & 0xff;
  let m = x & 0x7fffff;
  if (E === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  const e = E - 112; // rebias 127 -> 15
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m |= 0x800000;
    const shift = 14 - e, rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
    let h = m >>> shift;
    if (rem > half || (rem === half && h & 1)) h++;
    return sign | h;
  }
  let h = (e << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++; // a carry into the exponent is the correct rounding
  return sign | h;
}
/** f32 values -> packed f16 halves (padded to an even count, i.e. whole u32 words). */
function toHalf(v: ArrayLike<number>): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(v.length + (v.length & 1));
  for (let i = 0; i < v.length; i++) out[i] = f16Bits(v[i]!);
  return out;
}
function halfToF32(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

const pow2 = (n: number) => 2 ** Math.ceil(Math.log2(Math.max(1, n)));

/** Bump allocator for float offsets inside an activation buffer. */
class Layout {
  constructor(public size = 64) {} // [0] step counter, [1] position base
  take(n: number): number { const o = this.size; this.size += Math.ceil(n / 64) * 64; return o; }
}

export interface SynthOptions {
  seed?: number;
  /** Override the sampled flow noise ([maxFrames][32], already scaled). Testing only. */
  noise?: Float32Array<ArrayBuffer>;
}

export interface SynthResult {
  audio: Float32Array<ArrayBuffer>;
  frames: number;
  sampleRate: number;
}

export class MachBoomEngine {
  readonly sampleRate: number;
  private pipes = new Map<string, GPUComputePipeline>();
  private C!: GPUBuffer;
  private codes!: GPUBuffer;
  private W: Record<string, GPUBuffer> = {};
  private c: Record<string, number> = {};
  private voices = new Map<string, { rows: number; data: Float32Array<ArrayBuffer> }>();
  private ropeLM: GPUBuffer | null = null;
  private ropeDec: GPUBuffer | null = null;
  private ropeDecN = 0;
  private lm: null | {
    act: GPUBuffer; kv: GPUBuffer; Tcap: number; Pcap: number; Fcap: number;
    o: Record<string, number>; frame: Program; eosRead: GPUBuffer;
  } = null;
  private prefixVoice: string | null = null;
  private prefixRows = 0;
  private dec: null | { act: GPUBuffer; kv: GPUBuffer; o: Record<string, number>; progs: Map<number, Program> } = null;
  private chans = { c0: 0, conv: [] as { s: number; cin: number; co: number; rc: number }[] };
  private busy: Promise<unknown> = Promise.resolve();

  private constructor(readonly device: GPUDevice, private pack: MachPack, readonly tokenizer: UnigramTokenizer) {
    this.sampleRate = pack.cfg.dec.sample_rate;
  }

  static async requestDevice(): Promise<GPUDevice> {
    if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU is not available');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const lim = adapter.limits;
    return adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
        maxComputeWorkgroupStorageSize: Math.min(lim.maxComputeWorkgroupStorageSize, 16384),
      },
    });
  }

  static async create(weights: ArrayBuffer, tokenizerJson: any, device?: GPUDevice): Promise<MachBoomEngine> {
    const dev = device ?? (await MachBoomEngine.requestDevice());
    const e = new MachBoomEngine(dev, new MachPack(weights), new UnigramTokenizer(tokenizerJson));
    await e.init();
    return e;
  }

  get voiceNames(): string[] { return [...this.voices.keys()]; }

  addVoice(name: string, safetensors: ArrayBuffer) {
    this.voices.set(name, parseVoice(safetensors, this.pack.cfg.lm.hidden));
    if (this.prefixVoice === name) this.prefixVoice = null;
  }

  hasVoice(name: string) { return this.voices.has(name); }

  destroy() {
    this.lm?.act.destroy(); this.lm?.kv.destroy(); this.lm?.frame.destroy(); this.lm?.eosRead.destroy();
    this.dec?.act.destroy(); this.dec?.kv.destroy(); this.dec?.progs.forEach((p) => p.destroy());
    for (const b of Object.values(this.W)) b.destroy();
    this.C?.destroy(); this.codes?.destroy(); this.ropeLM?.destroy(); this.ropeDec?.destroy();
  }

  // ------------------------------------------------------------------ plumbing
  private pipe(name: string): GPUComputePipeline {
    let p = this.pipes.get(name);
    if (!p) {
      const k = KMAP[name]!;
      const module = this.device.createShaderModule({ code: kernelSource(k, BINDING_NAMES[name]!), label: name });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: name });
      this.pipes.set(name, p);
    }
    return p;
  }

  private add(prog: Program, name: string, params: Params, bufs: GPUBuffer[], wg: [number, number?, number?]) {
    const k: KernelDef = KMAP[name]!;
    for (const key of Object.keys(params)) {
      if (!k.fields.some(([n]) => n === key)) throw new Error(`${name}: unknown param ${key}`);
    }
    const bytes = Math.ceil((k.fields.length * 4) / 16) * 16;
    const ub = this.device.createBuffer({ size: bytes, usage: U().UNIFORM, mappedAtCreation: true });
    const dv = new DataView(ub.getMappedRange());
    k.fields.forEach(([n, t], i) => {
      const v = params[n] ?? (t === 'i32' ? -1 : 0);
      if (t === 'f32') dv.setFloat32(i * 4, v, true);
      else if (t === 'i32') dv.setInt32(i * 4, v, true);
      else {
        if (v < 0 || !Number.isInteger(v)) throw new Error(`${name}.${n} must be a u32, got ${v}`);
        dv.setUint32(i * 4, v, true);
      }
    });
    ub.unmap();
    prog.uniforms.push(ub);
    const pipe = this.pipe(name);
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [ub, ...bufs].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    prog.list.push({ pipe, bg, wg: [wg[0], wg[1] ?? 1, wg[2] ?? 1] });
  }

  private storage(floats: number, label?: string): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, floats * 4), usage: U().STORAGE | U().COPY_DST | U().COPY_SRC, label });
  }

  private upload(data: Float32Array<ArrayBuffer>, label?: string): GPUBuffer {
    const b = this.storage(data.length, label);
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  /** Storage for `halves` packed f16 values (whole u32 words). */
  private storage16(halves: number, label?: string): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(halves / 2) * 4), usage: U().STORAGE | U().COPY_DST | U().COPY_SRC, label });
  }

  private upload16(data: Float32Array<ArrayBuffer>, label?: string): GPUBuffer {
    const b = this.storage16(data.length, label);
    this.device.queue.writeBuffer(b, 0, toHalf(data));
    return b;
  }

  private bias(name: string): Float32Array<ArrayBuffer> | null {
    return this.pack.has(name + '.bias') ? this.pack.f32(name + '.bias') : null;
  }

  /** Dense [rows][cols] f32 of a non-trellis tensor (int8 or a f32 `.weight`). */
  private denseSmall(name: string): { data: Float32Array<ArrayBuffer>; rows: number; cols: number } {
    if (this.pack.has(name)) {
      const e = this.pack.entry(name);
      return { data: this.pack.small(name), rows: e.rows!, cols: e.cols! };
    }
    const e = this.pack.entry(name + '.weight');
    const data = this.pack.f32(name + '.weight');
    return { data, rows: e.shape[0]!, cols: data.length / e.shape[0]! };
  }

  // ------------------------------------------------------------------ init
  private async init() {
    const P = this.pack, cfg = P.cfg;
    for (const k of KERNELS) this.pipe(k.name);

    // whole data section as the code store (bulk + a padded 4-byte tail, no full copy)
    const data = P.u8.subarray(P.base);
    const bulk = data.length & ~3;
    this.codes = this.device.createBuffer({ size: Math.ceil(data.length / 4) * 4, usage: U().STORAGE | U().COPY_DST });
    if (bulk) this.device.queue.writeBuffer(this.codes, 0, data, 0, bulk);
    if (bulk < data.length) {
      const tail = new Uint8Array(4);
      tail.set(data.subarray(bulk));
      this.device.queue.writeBuffer(this.codes, bulk, tail);
    }
    const tl = new Float32Array(4096);
    tl.set(P.tlut10(3), 0);
    tl.set(P.tlut10(4), 2048);
    const tlut = this.upload(tl, 'tlut');
    const tlOff = (K: number) => (K === 3 ? 0 : K === 4 ? 2048 : (() => { throw new Error(`no tlut for K=${K}`); })());
    let scratchN = 0, stageN = 0;
    for (const e of Object.values(P.T)) {
      if (e.type !== 'trellis') continue;
      for (const ch of e.chunks!) scratchN = Math.max(scratchN, ch.m * ch.n);
      stageN = Math.max(stageN, e.shape.reduce((a, b) => a * b, 1));
    }
    const scratch = this.storage(scratchN, 'trellis-scratch');
    // Trellis output lands in an f32 stage (one tensor at a time) and is then
    // packed to f16 into its final buffer.
    const stage = this.storage(stageN, 'trellis-stage');
    const decode = new Program();
    /** Pack stage[0, n) into dest halves [dst, dst + n). */
    const pack = (dest: GPUBuffer, dst: number, n: number) => {
      if (dst % 2 || n % 2) throw new Error('f16 pack range must be even');
      const words = n / 2;
      this.add(decode, 'pack_f16', { n: words, src: 0, dst: dst / 2 }, [stage, dest], [Math.min(Math.ceil(words / 256), 65535)]);
    };

    const trellis = (name: string, dest: GPUBuffer, place: (ci: number, ch: { m: number; n: number }) => Params) => {
      const e = P.entry(name);
      if (e.type !== 'trellis') throw new Error(`${name} is not trellis`);
      e.chunks!.forEach((ch, ci) => {
        const prm: Params = {
          m: ch.m, n: ch.n, step: 2 * e.K!, ws: ch.ws, codes: ch.codes[0], su: ch.su[0], sv: ch.sv[0], tl: tlOff(e.K!),
          ...place(ci, ch),
        };
        this.add(decode, 'trellis_rows', prm, [this.codes, tlut, scratch], [ch.m]);
        this.add(decode, 'trellis_cols', prm, [this.codes, scratch, dest], [ch.n]);
      });
    };
    const consts = new ConstPool();
    const cv = (key: string, v: ArrayLike<number>) => { this.c[key] = consts.add(v); };

    /** A linear layer as a dense [out][in] buffer; several tensors may be stacked along rows. */
    const linear = (key: string, names: string[]): { rows: number; cols: number } => {
      const shapes = names.map((n) => {
        if (P.has(n) && P.entry(n).type === 'trellis') { const e = P.entry(n); return [e.shape[0]!, e.shape[1]!]; }
        const d = this.denseSmall(n); return [d.rows, d.cols];
      });
      const cols = shapes[0]![1]!;
      const rows = shapes.reduce((a, s) => a + s[0]!, 0);
      if (cols % 2) throw new Error(`${key}: odd width`);
      const buf = this.storage16(rows * cols, key);
      const biasAll = new Float32Array(rows);
      let anyBias = false, r0 = 0;
      names.forEach((n, i) => {
        if (shapes[i]![1] !== cols) throw new Error(`${n}: width mismatch`);
        if (P.has(n) && P.entry(n).type === 'trellis') {
          let cr = 0;
          trellis(n, stage, (_ci, ch) => {
            if (ch.n !== cols) throw new Error(`${n}: chunk splits columns`);
            const p = { mode: 0, base: cr * cols, rs: cols, cs: 1 };
            cr += ch.m;
            return p;
          });
          pack(buf, r0 * cols, shapes[i]![0]! * cols);
        } else {
          this.device.queue.writeBuffer(buf, r0 * cols * 2, toHalf(this.denseSmall(n).data));
        }
        const b = this.bias(n);
        if (b) { biasAll.set(b, r0); anyBias = true; }
        r0 += shapes[i]![0]!;
      });
      this.W[key] = buf;
      if (anyBias) cv(key + '.b', biasAll);
      return { rows, cols };
    };

    // ---------------- LM
    const lm = cfg.lm;
    for (let l = 0; l < lm.layers; l++) {
      const pre = `lm:model.layers.${l}.`;
      linear(`lm${l}.qkv`, ['q_proj', 'k_proj', 'v_proj'].map((s) => pre + 'self_attn.' + s));
      linear(`lm${l}.o`, [pre + 'self_attn.o_proj']);
      linear(`lm${l}.fc1`, [pre + 'mlp.fc1']);
      linear(`lm${l}.fc2`, [pre + 'mlp.fc2']);
      for (const [k, n] of [['ln1w', 'input_layernorm.weight'], ['ln1b', 'input_layernorm.bias'],
        ['ln2w', 'post_attention_layernorm.weight'], ['ln2b', 'post_attention_layernorm.bias']] as const) cv(`lm${l}.${k}`, P.f32(pre + n));
    }
    linear('lm.prompt', ['lm:model.audio_prompt_proj']);
    linear('lm.inlat', ['lm:model.audio_latent_proj']);
    {
      const cond = P.f32('const:cond');
      const b = this.bias('lm:model.audio_latent_proj');
      if (b) for (let i = 0; i < cond.length; i++) cond[i]! += b[i]!;
      cv('lm.inlat.b', cond);
    }
    cv('lm.normw', P.f32('lm:model.norm.weight'));
    cv('lm.normb', P.f32('lm:model.norm.bias'));
    cv('lm.eosw', P.f32('lm:eos_head.weight'));
    cv('lm.eosb', P.f32('lm:eos_head.bias'));
    // flow head
    linear('fl.inp', ['lm:flow_head.input_proj']);
    linear('fl.cond', ['lm:flow_head.cond_proj']);
    {
      const t = P.f32('const:flow_time');
      const b = this.bias('lm:flow_head.cond_proj');
      if (b) for (let i = 0; i < t.length; i++) t[i]! += b[i]!;
      cv('fl.cond.b', t);
    }
    const adaNames: string[] = [];
    for (let l = 0; l < lm.flow_layers; l++) {
      const pre = `lm:flow_head.layers.${l}.`;
      adaNames.push(pre + 'adaln_proj');
      linear(`fl${l}.fc1`, [pre + 'fc1']);
      linear(`fl${l}.fc2`, [pre + 'fc2']);
      cv(`fl${l}.nw`, P.f32(pre + 'norm.weight'));
      cv(`fl${l}.nb`, P.f32(pre + 'norm.bias'));
    }
    adaNames.push('lm:flow_head.final_layer.adaln_proj');
    const ada = linear('fl.ada', adaNames);
    this.c['fl.adaRows'] = ada.rows;
    linear('fl.proj', ['lm:flow_head.final_layer.proj']);
    const mean = P.f32('lm:latent_mean'), std = P.f32('lm:latent_std');

    // ---------------- decoder
    const d = cfg.dec;
    {
      const { data, rows, cols } = this.denseSmall('dec:latent_proj');
      const w = new Float32Array(rows * cols), b = new Float32Array(rows);
      const own = this.bias('dec:latent_proj');
      for (let o = 0; o < rows; o++) {
        let s = own ? own[o]! : 0;
        for (let i = 0; i < cols; i++) { w[o * cols + i] = data[o * cols + i]! * std[i]!; s += data[o * cols + i]! * mean[i]!; }
        b[o] = s;
      }
      this.W['dec.lat'] = this.upload16(w, 'dec.lat');
      cv('dec.lat.b', b);
      cv('dec.up', this.denseSmall('dec:decoder.upsample.conv').data);
    }
    for (let l = 0; l < d.layers; l++) {
      const pre = `dec:decoder.decoder_transformer.layers.${l}.`;
      linear(`dec${l}.qkv`, ['q_proj', 'k_proj', 'v_proj'].map((s) => pre + 'self_attn.' + s));
      linear(`dec${l}.o`, [pre + 'self_attn.o_proj']);
      linear(`dec${l}.fc1`, [pre + 'mlp.fc1']);
      linear(`dec${l}.fc2`, [pre + 'mlp.fc2']);
      for (const [k, n] of [['ln1w', 'input_layernorm.weight'], ['ln1b', 'input_layernorm.bias'],
        ['ln2w', 'post_attention_layernorm.weight'], ['ln2b', 'post_attention_layernorm.bias'],
        ['lsa', 'self_attn_layer_scale.scale'], ['lsm', 'mlp_layer_scale.scale']] as const) cv(`dec${l}.${k}`, P.f32(pre + n));
    }
    /** Causal conv as [out][K*in] (row k*in+i of the im2col window). */
    const conv = (key: string, name: string): { K: number; cin: number; co: number } => {
      let K: number, cin: number, co: number;
      if (P.has(name)) {
        const e = P.entry(name);
        [co, cin, K] = [e.shape[0]!, e.shape[1]!, e.shape[2] ?? 1];
        const buf = this.storage16(co * K * cin, key);
        if (e.kind === 'taps') trellis(name, stage, (ci) => ({ mode: 0, base: ci * cin, rs: K * cin, cs: 1 }));
        else if (K === 1) trellis(name, stage, () => ({ mode: 0, base: 0, rs: cin, cs: 1 }));
        else throw new Error(`${name}: rows-coded conv with K=${K} is not supported`);
        pack(buf, 0, co * K * cin);
        this.W[key] = buf;
      } else {
        const e = P.entry(name + '.weight');
        [co, cin, K] = [e.shape[0]!, e.shape[1]!, e.shape[2]!];
        const w = P.f32(name + '.weight'), out = new Float32Array(co * K * cin);
        for (let o = 0; o < co; o++) for (let i = 0; i < cin; i++) for (let k = 0; k < K; k++)
          out[o * K * cin + k * cin + i] = w[(o * cin + i) * K + k]!;
        this.W[key] = this.upload16(out, key);
      }
      const b = this.bias(name);
      cv(key + '.b', b ?? new Float32Array(co));
      return { K, cin, co };
    };
    /** Transposed conv (K = 2s) as [s*co][2*cin]: column block 0 multiplies x[t-1] (taps k+s), block 1 x[t] (taps k). */
    const convT = (key: string, name: string): { s: number; cin: number; co: number } => {
      let cin: number, co: number, K: number;
      if (P.has(name)) {
        const e = P.entry(name);
        [cin, co, K] = [e.shape[0]!, e.shape[1]!, e.shape[2]!];
        const s = K / 2;
        const buf = this.storage16(s * co * 2 * cin, key);
        if (e.kind === 'taps') {
          trellis(name, stage, (k) => ({ mode: 0, base: (k % s) * co * 2 * cin + (k < s ? cin : 0), rs: 2 * cin, cs: 1 }));
        } else {
          trellis(name, stage, () => ({ mode: 2, base: 0, K, s, co, cin }));
        }
        pack(buf, 0, s * co * 2 * cin);
        this.W[key] = buf;
      } else {
        const e = P.entry(name + '.weight');
        [cin, co, K] = [e.shape[0]!, e.shape[1]!, e.shape[2]!];
        const s = K / 2, w = P.f32(name + '.weight'), out = new Float32Array(s * co * 2 * cin);
        for (let i = 0; i < cin; i++) for (let o = 0; o < co; o++) for (let k = 0; k < K; k++)
          out[((k % s) * co + o) * 2 * cin + (k < s ? cin : 0) + i] = w[(i * co + o) * K + k]!;
        this.W[key] = this.upload16(out, key);
      }
      const s = K / 2;
      const b = this.bias(name) ?? new Float32Array(co);
      const tb = new Float32Array(s * co);
      for (let k = 0; k < s; k++) tb.set(b, k * co);
      cv(key + '.b', tb);
      return { s, cin, co };
    };
    const c0 = conv('sea0', SEA + '0.conv');
    if (c0.K !== d.kernel) throw new Error('unexpected first conv kernel');
    this.chans.c0 = c0.co;
    for (let j = 0; j < d.ratios.length; j++) {
      const li = 1 + 3 * j;
      const t = convT(`seaT${j}`, `${SEA}${li + 1}.conv`);
      const b1 = conv(`seaB1_${j}`, `${SEA}${li + 2}.block.1.conv`);
      const b3 = conv(`seaB3_${j}`, `${SEA}${li + 2}.block.3.conv`);
      if (t.s !== d.ratios[j] || b1.cin !== t.co || b3.co !== t.co || b1.K !== d.res_kernel) throw new Error(`seanet stage ${j} shape`);
      this.chans.conv.push({ s: t.s, cin: t.cin, co: t.co, rc: b1.co });
    }
    const last = conv('seaLast', `${SEA}${2 + 3 * d.ratios.length}.conv`);
    if (last.co !== 1 || last.K !== d.last_kernel) throw new Error('unexpected last conv');

    this.C = consts.build(this.device);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    decode.encode(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    decode.destroy();
    scratch.destroy();
    stage.destroy();
    tlut.destroy();
    this.codes.destroy();
    // Only the token embedding and the BOS row are read on the CPU after this;
    // drop the rest of the weight file so its 17 MB ArrayBuffer can be collected.
    P.retain(['lm:model.embed_tokens', 'const:bos']);
  }

  // ------------------------------------------------------------------ LM graph
  private ensureLM(T: number, Ptot: number, F: number) {
    const cur = this.lm;
    if (cur && cur.Tcap >= T && cur.Pcap >= Ptot && cur.Fcap >= F) return;
    const Tcap = Math.max(pow2(T), cur?.Tcap ?? 0, 256);
    const Pcap = Math.max(pow2(Ptot), cur?.Pcap ?? 0, 1024);
    const Fcap = Math.max(pow2(F + 2), cur?.Fcap ?? 0, 512);
    if (cur) { cur.act.destroy(); cur.kv.destroy(); cur.frame.destroy(); cur.eosRead.destroy(); }
    this.ropeLM?.destroy();
    this.ropeLM = this.upload(ropeTable(this.pack.cfg.lm.rope_theta, Pcap), 'rope-lm');
    const cfg = this.pack.cfg.lm, D = cfg.hidden, L = new Layout();
    const o: Record<string, number> = {
      hfin: L.take(D), x: L.take(Tcap * D), h: L.take(Tcap * D), qkv: L.take(Tcap * 3 * D), att: L.take(Tcap * D),
      u: L.take(Tcap * cfg.ffn), eos: L.take(Fcap), noise: L.take(Fcap * cfg.latent_dim), lats: L.take(Fcap * cfg.latent_dim),
      sc: L.take(D), ada: L.take(this.c['fl.adaRows']!), fx: L.take(D), fu: L.take(D),
    };
    const KV = new Layout(0); // offsets in f16 halves
    for (let l = 0; l < cfg.layers; l++) { o[`K${l}`] = KV.take(Pcap * D); o[`V${l}`] = KV.take(Pcap * D); }
    const act = this.storage(L.size, 'lm-act');
    const kv = this.storage16(KV.size, 'lm-kv');
    const eosRead = this.device.createBuffer({ size: Fcap * 4, usage: U().MAP_READ | U().COPY_DST });
    this.lm = { act, kv, Tcap, Pcap, Fcap, o, frame: new Program(), eosRead };
    this.buildFrame();
    this.prefixVoice = null;
  }

  private lmLayers(prog: Program, T: number, posB: number) {
    const cfg = this.pack.cfg.lm, D = cfg.hidden, H = cfg.heads, { act, kv, o, Pcap } = this.lm!, c = this.c;
    const step = T === 1 && posB < 0; // single-row autoregressive step at the dynamic position
    for (let l = 0; l < cfg.layers; l++) {
      const W = (k: string) => this.W[`lm${l}.${k}`]!;
      if (step) {
        this.add(prog, 'gemv', { m: 3 * D, n: D, xOff: o.x!, yOff: o.qkv!, pro: 1, lnW: c[`lm${l}.ln1w`]!, lnB: c[`lm${l}.ln1b`]!, eps: cfg.ln_eps },
          [W('qkv'), act, this.C], [(3 * D) / 8]);
      } else {
        this.add(prog, 'layernorm', { n: D, xOff: o.x!, xS: D, yOff: o.h!, yS: D, w: c[`lm${l}.ln1w`]!, b: c[`lm${l}.ln1b`]!, eps: cfg.ln_eps }, [act, this.C], [T]);
        this.gemm(prog, act, W('qkv'), { M: T, N: 3 * D, K: D, aOff: o.h!, lda: D, cOff: o.qkv!, ldc: 3 * D });
      }
      const pos = step ? { posB: 0, posFS: 1, dyn: 1 } : { posB, posFS: 0, dyn: 0 };
      this.add(prog, 'rope_kv', { T, H, qOff: o.qkv!, qS: 3 * D, ...pos, kOff: o[`K${l}`]!, vOff: o[`V${l}`]!, R: Pcap }, [act, this.ropeLM!, kv], [Math.ceil((T * H * 16) / 256)]);
      this.add(prog, 'attention', { H, qOff: o.qkv!, qS: 3 * D, ...pos, kOff: o[`K${l}`]!, vOff: o[`V${l}`]!, R: Pcap, win: 0, oOff: o.att!, oS: D, scale: 1 / Math.sqrt(cfg.head_dim) }, [act, kv], [T, H]);
      if (step) {
        this.add(prog, 'gemv', { m: D, n: D, xOff: o.att!, yOff: o.x!, epi: 1 }, [W('o'), act, this.C], [D / 8]);
        this.add(prog, 'gemv', { m: cfg.ffn, n: D, xOff: o.x!, yOff: o.u!, pro: 1, lnW: c[`lm${l}.ln2w`]!, lnB: c[`lm${l}.ln2b`]!, eps: cfg.ln_eps, actf: 2 },
          [W('fc1'), act, this.C], [cfg.ffn / 8]);
        this.add(prog, 'gemv', { m: D, n: cfg.ffn, xOff: o.u!, yOff: o.x!, epi: 1 }, [W('fc2'), act, this.C], [D / 8]);
      } else {
        this.gemm(prog, act, W('o'), { M: T, N: D, K: D, aOff: o.att!, lda: D, cOff: o.x!, ldc: D, res: o.x! });
        this.add(prog, 'layernorm', { n: D, xOff: o.x!, xS: D, yOff: o.h!, yS: D, w: c[`lm${l}.ln2w`]!, b: c[`lm${l}.ln2b`]!, eps: cfg.ln_eps }, [act, this.C], [T]);
        this.gemm(prog, act, W('fc1'), { M: T, N: cfg.ffn, K: D, aOff: o.h!, lda: D, cOff: o.u!, ldc: cfg.ffn, actf: 2 });
        this.gemm(prog, act, W('fc2'), { M: T, N: D, K: cfg.ffn, aOff: o.u!, lda: cfg.ffn, cOff: o.x!, ldc: D, res: o.x! });
      }
    }
  }

  private gemm(prog: Program, act: GPUBuffer, W: GPUBuffer, p: Params) {
    this.add(prog, 'gemm', p, [W, act, this.C], [Math.ceil(p.N! / 64), Math.ceil(p.M! / 64)]);
  }

  private lmFinal(prog: Program, row: number, add: number, inc: number) {
    const { act, o } = this.lm!, c = this.c, D = this.pack.cfg.lm.hidden;
    this.add(prog, 'lm_final', { xOff: o.x! + row * D, hOff: o.hfin!, w: c['lm.normw']!, b: c['lm.normb']!, eps: this.pack.cfg.lm.ln_eps,
      ew: c['lm.eosw']!, eb: c['lm.eosb']!, eosOff: o.eos!, add, inc }, [act, this.C], [1]);
  }

  /** One autoregressive frame: flow head on hfin/noise[f] -> lats[f], then LM step -> hfin, eos[f+1]. */
  private buildFrame() {
    const cfg = this.pack.cfg.lm, D = cfg.hidden, LD = cfg.latent_dim, { act, o, frame } = this.lm!, c = this.c;
    const W = (k: string) => this.W[k]!;
    this.add(frame, 'gemv', { m: D, n: D, xOff: o.hfin!, yOff: o.sc!, bias: c['fl.cond.b']!, actf: 1 }, [W('fl.cond'), act, this.C], [D / 8]);
    this.add(frame, 'gemv', { m: D, n: LD, xOff: o.noise!, xFS: LD, yOff: o.fx!, bias: c['fl.inp.b'] ?? -1 }, [W('fl.inp'), act, this.C], [D / 8]);
    const AR = c['fl.adaRows']!;
    this.add(frame, 'gemv', { m: AR, n: D, xOff: o.sc!, yOff: o.ada!, bias: c['fl.ada.b'] ?? -1 }, [W('fl.ada'), act, this.C], [Math.ceil(AR / 8)]);
    for (let l = 0; l < cfg.flow_layers; l++) {
      const a = o.ada! + l * 3 * D;
      this.add(frame, 'gemv', { m: D, n: D, xOff: o.fx!, yOff: o.fu!, pro: 2, lnW: c[`fl${l}.nw`]!, lnB: c[`fl${l}.nb`]!, eps: cfg.flow_eps,
        shOff: a, scOff: a + D, bias: c[`fl${l}.fc1.b`] ?? -1, actf: 1 }, [W(`fl${l}.fc1`), act, this.C], [D / 8]);
      this.add(frame, 'gemv', { m: D, n: D, xOff: o.fu!, yOff: o.fx!, bias: c[`fl${l}.fc2.b`] ?? -1, epi: 2, gOff: a + 2 * D },
        [W(`fl${l}.fc2`), act, this.C], [D / 8]);
    }
    const af = o.ada! + cfg.flow_layers * 3 * D;
    this.add(frame, 'gemv', { m: LD, n: D, xOff: o.fx!, yOff: o.lats!, yFS: LD, pro: 2, lnW: -1, lnB: -1, eps: cfg.flow_eps, shOff: af, scOff: af + D,
      bias: c['fl.proj.b'] ?? -1, epi: 3, rOff: o.noise!, rFS: LD }, [W('fl.proj'), act, this.C], [LD / 8]);
    this.add(frame, 'gemv', { m: D, n: LD, xOff: o.lats!, xFS: LD, yOff: o.x!, bias: c['lm.inlat.b']! }, [W('lm.inlat'), act, this.C], [D / 8]);
    this.lmLayers(frame, 1, -1);
    this.lmFinal(frame, 0, 1, 1);
  }

  private setCounters(act: GPUBuffer, step: number, base: number) {
    this.device.queue.writeBuffer(act, 0, new Float32Array([step, base]));
  }

  private runProgram(prog: Program, extra?: (enc: GPUCommandEncoder) => void) {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    prog.encode(pass);
    pass.end();
    extra?.(enc);
    this.device.queue.submit([enc.finish()]);
  }

  private voicePrefix(name: string) {
    const v = this.voices.get(name);
    if (!v) throw new Error(`unknown voice ${name}`);
    const { act, o } = this.lm!, D = this.pack.cfg.lm.hidden;
    this.device.queue.writeBuffer(act, o.h! * 4, v.data);
    this.setCounters(act, 0, 0);
    const prog = new Program();
    this.gemm(prog, act, this.W['lm.prompt']!, { M: v.rows, N: D, K: D, aOff: o.h!, lda: D, cOff: o.x!, ldc: D, bias: this.c['lm.prompt.b'] ?? -1 });
    this.lmLayers(prog, v.rows, 0);
    this.runProgram(prog);
    this.device.queue.onSubmittedWorkDone().then(() => prog.destroy());
    this.prefixVoice = name;
    this.prefixRows = v.rows;
  }

  // ------------------------------------------------------------------ decoder graph
  private ensureDec(rowsNeeded: number) {
    const d = this.pack.cfg.dec;
    if (!this.ropeDec || this.ropeDecN < rowsNeeded) {
      const n = Math.max(pow2(rowsNeeded), 8192);
      this.ropeDec?.destroy();
      this.ropeDec = this.upload(ropeTable(d.rope_theta, n), 'rope-dec');
      this.ropeDecN = n;
      this.dec?.progs.forEach((p) => p.destroy());
      this.dec?.progs.clear();
    }
    if (this.dec) return;
    const D = d.hidden, R = 2 * 256, T0 = 256, L = new Layout(), LD = this.pack.cfg.lm.latent_dim, c0 = this.chans.c0;
    const rk = d.res_kernel, nS = this.chans.conv.length;
    // Persistent rows: inputs, the carried conv/transformer history (kept in
    // small stores and restored into the head of each transient region per
    // chunk), and the output wave.
    const o: Record<string, number> = { latin: L.take(16 * LD), xl: L.take(17 * D), xb0: L.take((6 + T0) * D) };
    this.chans.conv.forEach((s, j) => { o[`hin${j}`] = L.take(s.cin); o[`hpre${j}`] = L.take((rk - 1) * s.co); });
    const lc = this.chans.conv[nS - 1]!.co;
    o.hlast = L.take((d.last_kernel - 1) * lc);
    // Transient regions, reused across phases:
    //   P  : transformer scratch (h, qkv, att, u), then each stage's pre_j + r_j
    //   Q0 : post_j for even j;  Q1 : B0 (sea0 output), then post_j for odd j
    const tr = new Layout(0);
    const rel: Record<string, number> = { h: tr.take(T0 * D), qkv: tr.take(T0 * 3 * D), att: tr.take(T0 * D), u: tr.take(T0 * d.ffn) };
    let pN = tr.size, q0N = 0, q1N = (1 + T0) * c0, T = T0;
    const stageRel: { pre: number; r: number }[] = [];
    this.chans.conv.forEach((s, j) => {
      T *= s.s;
      const hist = j === nS - 1 ? d.last_kernel - 1 : 1;
      const st = new Layout(0);
      stageRel.push({ pre: st.take((rk - 1 + T) * s.co), r: st.take(T * s.rc) });
      pN = Math.max(pN, st.size);
      const post = (hist + T) * s.co;
      if (j % 2 === 0) q0N = Math.max(q0N, post); else q1N = Math.max(q1N, post);
    });
    const Pb = L.take(pN), Q0 = L.take(q0N), Q1 = L.take(q1N);
    for (const k of Object.keys(rel)) o[k] = Pb + rel[k]!;
    o.B0 = Q1;
    stageRel.forEach((r, j) => { o[`pre${j}`] = Pb + r.pre; o[`r${j}`] = Pb + r.r; o[`post${j}`] = j % 2 === 0 ? Q0 : Q1; });
    o.wav = L.take(T);
    const KV = new Layout(0); // f16 halves
    for (let l = 0; l < d.layers; l++) { o[`K${l}`] = KV.take(R * D); o[`V${l}`] = KV.take(R * D); }
    this.dec = { act: this.storage(L.size, 'dec-act'), kv: this.storage16(KV.size, 'dec-kv'), o, progs: new Map() };
  }

  private decProgram(F: number): Program {
    const dd = this.dec!;
    let prog = dd.progs.get(F);
    if (prog) return prog;
    prog = new Program();
    const d = this.pack.cfg.dec, D = d.hidden, H = d.heads, { act, kv, o } = dd, c = this.c, LD = this.pack.cfg.lm.latent_dim;
    const st = d.latent_stride, T0 = F * st, R = 512;
    this.gemm(prog, act, this.W['dec.lat']!, { M: F, N: D, K: LD, aOff: o.latin!, lda: LD, cOff: o.xl! + D, ldc: D, bias: c['dec.lat.b']! });
    this.add(prog, 'upsample', { F, xOff: o.xl!, w: c['dec.up']!, oOff: o.xb0! + 6 * D }, [act, this.C], [Math.ceil((T0 * D) / 256)]);
    this.add(prog, 'copy', { n: D, src: o.xl! + F * D, dst: o.xl! }, [act], [D / 256]);
    const X = o.xb0! + (d.kernel - 1) * D;
    for (let l = 0; l < d.layers; l++) {
      const W = (k: string) => this.W[`dec${l}.${k}`]!;
      this.add(prog, 'layernorm', { n: D, xOff: X, xS: D, yOff: o.h!, yS: D, w: c[`dec${l}.ln1w`]!, b: c[`dec${l}.ln1b`]!, eps: d.eps }, [act, this.C], [T0]);
      this.gemm(prog, act, W('qkv'), { M: T0, N: 3 * D, K: D, aOff: o.h!, lda: D, cOff: o.qkv!, ldc: 3 * D, bias: c[`dec${l}.qkv.b`] ?? -1 });
      const pos = { posB: 0, posFS: 16 * st, dyn: 0 };
      this.add(prog, 'rope_kv', { T: T0, H, qOff: o.qkv!, qS: 3 * D, ...pos, kOff: o[`K${l}`]!, vOff: o[`V${l}`]!, R }, [act, this.ropeDec!, kv], [Math.ceil((T0 * H * 16) / 256)]);
      this.add(prog, 'attention', { H, qOff: o.qkv!, qS: 3 * D, ...pos, kOff: o[`K${l}`]!, vOff: o[`V${l}`]!, R, win: d.window, oOff: o.att!, oS: D,
        scale: 1 / Math.sqrt(d.head_dim) }, [act, kv], [T0, H]);
      this.gemm(prog, act, W('o'), { M: T0, N: D, K: D, aOff: o.att!, lda: D, cOff: X, ldc: D, bias: c[`dec${l}.o.b`] ?? -1, scl: c[`dec${l}.lsa`]!, res: X });
      this.add(prog, 'layernorm', { n: D, xOff: X, xS: D, yOff: o.h!, yS: D, w: c[`dec${l}.ln2w`]!, b: c[`dec${l}.ln2b`]!, eps: d.eps }, [act, this.C], [T0]);
      this.gemm(prog, act, W('fc1'), { M: T0, N: d.ffn, K: D, aOff: o.h!, lda: D, cOff: o.u!, ldc: d.ffn, bias: c[`dec${l}.fc1.b`] ?? -1, actf: 2 });
      this.gemm(prog, act, W('fc2'), { M: T0, N: D, K: d.ffn, aOff: o.u!, lda: d.ffn, cOff: X, ldc: D, bias: c[`dec${l}.fc2.b`] ?? -1, scl: c[`dec${l}.lsm`]!, res: X });
    }
    const c0 = this.chans.c0;
    this.gemm(prog, act, this.W.sea0!, { M: T0, N: c0, K: d.kernel * D, aOff: o.xb0!, lda: D, cOff: o.B0! + c0, ldc: c0, bias: c['sea0.b']! });
    this.add(prog, 'copy', { n: (d.kernel - 1) * D, src: o.xb0! + T0 * D, dst: o.xb0! }, [act], [Math.ceil(((d.kernel - 1) * D) / 256)]);
    let inp = o.B0!, Tin = T0;
    const rk = d.res_kernel;
    const copy = (n: number, src: number, dst: number) => this.add(prog, 'copy', { n, src, dst }, [act], [Math.ceil(n / 256)]);
    this.chans.conv.forEach((s, j) => {
      const Tout = Tin * s.s;
      const pre = o[`pre${j}`]!, r = o[`r${j}`]!, post = o[`post${j}`]!;
      const hist = j === this.chans.conv.length - 1 ? d.last_kernel - 1 : 1;
      // inp/pre live in reused regions: restore their history rows from the stores first
      copy(s.cin, o[`hin${j}`]!, inp);
      copy((rk - 1) * s.co, o[`hpre${j}`]!, pre);
      this.gemm(prog, act, this.W[`seaT${j}`]!, { M: Tin, N: s.s * s.co, K: 2 * s.cin, aOff: inp, lda: s.cin, eluIn: 1, cOff: pre + (rk - 1) * s.co, ldc: s.s * s.co, bias: c[`seaT${j}.b`]! });
      copy(s.cin, inp + Tin * s.cin, o[`hin${j}`]!);
      this.gemm(prog, act, this.W[`seaB1_${j}`]!, { M: Tout, N: s.rc, K: rk * s.co, aOff: pre, lda: s.co, eluIn: 1, cOff: r, ldc: s.rc, bias: c[`seaB1_${j}.b`]! });
      this.gemm(prog, act, this.W[`seaB3_${j}`]!, { M: Tout, N: s.co, K: s.rc, aOff: r, lda: s.rc, eluIn: 1, cOff: post + hist * s.co, ldc: s.co,
        bias: c[`seaB3_${j}.b`]!, res: pre + (rk - 1) * s.co });
      copy((rk - 1) * s.co, pre + Tout * s.co, o[`hpre${j}`]!);
      inp = post; Tin = Tout;
    });
    const lc = this.chans.conv[this.chans.conv.length - 1]!.co, lk = d.last_kernel;
    copy((lk - 1) * lc, o.hlast!, inp);
    this.gemm(prog, act, this.W.seaLast!, { M: Tin, N: 1, K: lk * lc, aOff: inp, lda: lc, eluIn: 1, cOff: o.wav!, ldc: 1, bias: c['seaLast.b']! });
    copy((lk - 1) * lc, inp + Tin * lc, o.hlast!);
    dd.progs.set(F, prog);
    return prog;
  }

  /** Zero every carried history row so a new utterance starts from silence. */
  private resetDec(enc: GPUCommandEncoder) {
    const d = this.pack.cfg.dec, { act, o } = this.dec!, D = d.hidden;
    const clear = (off: number, n: number) => enc.clearBuffer(act, off * 4, n * 4);
    clear(o.xl!, D);
    clear(o.xb0!, (d.kernel - 1) * D);
    this.chans.conv.forEach((s, j) => {
      clear(o[`hin${j}`]!, s.cin);
      clear(o[`hpre${j}`]!, (d.res_kernel - 1) * s.co);
    });
    clear(o.hlast!, (d.last_kernel - 1) * this.chans.conv[this.chans.conv.length - 1]!.co);
  }

  // ------------------------------------------------------------------ synthesis
  /** Text (already split into a sentence-sized chunk) -> 24 kHz mono audio. */
  synthesize(text: string, voice: string, opts: SynthOptions = {}): Promise<SynthResult> {
    const run = this.busy.then(() => this.synthNow(text, voice, opts));
    this.busy = run.catch(() => {});
    return run;
  }

  /** Normalized text -> token ids and word count (exposed for testing). */
  textIds(text: string, frontend = true) {
    return this.tokenizer.textIds(frontend ? normalize(text) : text);
  }

  private async synthNow(text: string, voice: string, opts: SynthOptions): Promise<SynthResult> {
    const P = this.pack, cfg = P.cfg, g = cfg.gen, D = cfg.lm.hidden, LD = cfg.lm.latent_dim;
    const { ids, numWords } = this.textIds(text);
    if (!ids.length) return { audio: new Float32Array(0), frames: 0, sampleRate: this.sampleRate };
    const v = this.voices.get(voice);
    if (!v) throw new Error(`unknown voice ${voice}`);
    const nt = ids.length;
    const maxF = Math.ceil((nt / 3 + 2) * cfg.lm.frame_rate);
    this.ensureLM(Math.max(v.rows, nt + 1), v.rows + nt + 1 + maxF, maxF);
    if (this.prefixVoice !== voice) this.voicePrefix(voice);
    const Pv = this.prefixRows, lm = this.lm!, { act, o } = lm;

    // text prefill
    const x = new Float32Array((nt + 1) * D);
    const row = new Float32Array(D);
    ids.forEach((id, i) => x.set(P.embedRow(id, row), i * D));
    x.set(P.f32('const:bos'), nt * D);
    this.device.queue.writeBuffer(act, o.x! * 4, x);
    const noise = opts.noise ?? gaussian(maxF * LD, opts.seed ?? 0, Math.sqrt(g.temperature));
    this.device.queue.writeBuffer(act, o.noise! * 4, noise.subarray(0, maxF * LD));
    this.setCounters(act, 0, Pv + nt + 1);
    const pre = new Program();
    this.lmLayers(pre, nt + 1, Pv);
    this.lmFinal(pre, nt, 0, 0);
    this.runProgram(pre);

    // autoregressive frames in speculative batches, checking eos between batches
    const fae = numWords <= g.short_text_max_words ? g.frames_after_eos_short_text : g.frames_after_eos;
    let done = 0, frames = -1, batch = 8;
    while (frames < 0) {
      const B = Math.min(batch, maxF - done);
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let i = 0; i < B; i++) lm.frame.encode(pass);
      pass.end();
      done += B;
      enc.copyBufferToBuffer(act, o.eos! * 4, lm.eosRead, 0, (done + 1) * 4);
      this.device.queue.submit([enc.finish()]);
      await lm.eosRead.mapAsync(GPUMapMode.READ, 0, (done + 1) * 4);
      const eos = new Float32Array(lm.eosRead.getMappedRange(0, (done + 1) * 4).slice(0));
      lm.eosRead.unmap();
      let remaining = -1;
      for (let f = 0; f < done; f++) {
        if (remaining < 0 && eos[f]! > g.eos_threshold) remaining = fae;
        if (remaining === 0 || f === maxF - 1) { frames = f + 1; break; }
        if (remaining > 0) remaining--;
      }
      batch = Math.min(batch * 2, 32);
    }
    this.device.queue.onSubmittedWorkDone().then(() => pre.destroy());
    const audio = await this.decode(lm.act, o.lats!, frames);
    return { audio, frames, sampleRate: this.sampleRate };
  }

  /** Decode `frames` normalized latents starting at float offset `src` of buffer `srcBuf`. */
  async decode(srcBuf: GPUBuffer, src: number, frames: number): Promise<Float32Array<ArrayBuffer>> {
    const d = this.pack.cfg.dec, LD = this.pack.cfg.lm.latent_dim, st = d.latent_stride;
    const spf = st * d.ratios.reduce((a, b) => a * b, 1);
    this.ensureDec(Math.ceil(frames / 16) * 16 * st + 16);
    const { act, o } = this.dec!;
    const out = this.device.createBuffer({ size: Math.max(4, frames * spf * 4), usage: U().MAP_READ | U().COPY_DST });
    for (let c = 0; c * 16 < frames; c++) {
      const F = Math.min(16, frames - c * 16);
      const prog = this.decProgram(F);
      this.setCounters(act, c, 0);
      const enc = this.device.createCommandEncoder();
      if (c === 0) this.resetDec(enc);
      enc.copyBufferToBuffer(srcBuf, (src + c * 16 * LD) * 4, act, o.latin! * 4, F * LD * 4);
      const pass = enc.beginComputePass();
      prog.encode(pass);
      pass.end();
      enc.copyBufferToBuffer(act, o.wav! * 4, out, c * 16 * spf * 4, F * spf * 4);
      this.device.queue.submit([enc.finish()]);
    }
    await out.mapAsync(GPUMapMode.READ);
    const audio = new Float32Array(out.getMappedRange().slice(0));
    out.unmap();
    out.destroy();
    return audio;
  }

  /** Test hook: decode host latents (normalized, [F][32]). */
  async decodeLatents(lats: Float32Array<ArrayBuffer>): Promise<Float32Array<ArrayBuffer>> {
    const b = this.upload(lats, 'lats-in');
    try { return await this.decode(b, 0, lats.length / this.pack.cfg.lm.latent_dim); } finally { b.destroy(); }
  }

  /** Test hook: read back a float range of the LM activation buffer. */
  async readLM(key: string, n: number): Promise<Float32Array<ArrayBuffer>> {
    const { act, o } = this.lm!;
    const rb = this.device.createBuffer({ size: n * 4, usage: U().MAP_READ | U().COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(act, o[key]! * 4, rb, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return r;
  }

  /** Test hook: read back (and unpack) the first n values of a dense f16 weight buffer. */
  async readWeight(key: string, n: number): Promise<Float32Array<ArrayBuffer>> {
    const src = this.W[key]!, bytes = Math.ceil(n / 2) * 4;
    const rb = this.device.createBuffer({ size: bytes, usage: U().MAP_READ | U().COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rb, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const h = new Uint16Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    const r = new Float32Array(n);
    for (let i = 0; i < n; i++) r[i] = halfToF32(h[i]!);
    return r;
  }
}

