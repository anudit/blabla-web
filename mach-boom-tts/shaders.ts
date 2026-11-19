/**
 * WGSL kernels for Mach Boom. Every kernel takes one uniform param block
 * (binding 0) whose WGSL struct is generated from the same field list the
 * engine packs, so the two can never drift apart.
 *
 * Activations for a graph live in one read_write f32 buffer (`act`); element 0
 * of it is the graph's step counter (stored as an f32 integer), which lets a
 * whole batch of autoregressive frames be encoded without CPU round trips.
 * Element 1 is a position base that rope_kv/attention add when `dyn` is set.
 *
 * Dense weights and the attention KV caches are stored as packed f16 pairs in
 * `array<u32>` (unpack2x16float / pack2x16float), which halves their memory
 * without requiring the optional shader-f16 feature. Math stays in f32.
 */

export type FieldType = 'u32' | 'i32' | 'f32';
export type Fields = readonly (readonly [string, FieldType])[];

export interface KernelDef {
  name: string;
  fields: Fields;
  /** Storage bindings after the uniform, in binding order. */
  bindings: readonly ('read' | 'rw' | 'read_u32' | 'rw_u32')[];
  body: string;
}

function structOf(fields: Fields): string {
  return `struct Params {\n${fields.map(([n, t]) => `  ${n}: ${t},`).join('\n')}\n}\n@group(0) @binding(0) var<uniform> p: Params;\n`;
}

export function kernelSource(k: KernelDef, bindingNames: readonly string[]): string {
  const binds = k.bindings
    .map((b, i) => {
      const name = bindingNames[i];
      if (b === 'rw_u32') return `@group(0) @binding(${i + 1}) var<storage, read_write> ${name}: array<u32>;`;
      if (b === 'read_u32') return `@group(0) @binding(${i + 1}) var<storage, read> ${name}: array<u32>;`;
      if (b === 'read') return `@group(0) @binding(${i + 1}) var<storage, read> ${name}: array<f32>;`;
      return `@group(0) @binding(${i + 1}) var<storage, read_write> ${name}: array<f32>;`;
    })
    .join('\n');
  return `${structOf(k.fields)}${binds}\n${ACT_FNS}\n${k.body}`;
}

const ACT_FNS = /* wgsl */ `
fn erf_as(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, x >= 0.0);
}
fn gelu_f(v: f32) -> f32 { return 0.5 * v * (1.0 + erf_as(clamp(v * 0.70710678118654752, -6.0, 6.0))); }
fn silu_f(v: f32) -> f32 { return v / (1.0 + exp(clamp(-v, -80.0, 80.0))); }
fn elu_f(v: f32) -> f32 { if (v > 0.0) { return v; } return exp(max(v, -30.0)) - 1.0; }
fn act_f(v: f32, k: u32) -> f32 {
  if (k == 1u) { return silu_f(v); }
  if (k == 2u) { return gelu_f(v); }
  if (k == 3u) { return elu_f(v); }
  return v;
}
`;

/** Shared 256-wide workgroup sum (call from uniform control flow). */
const WSUM = /* wgsl */ `
var<workgroup> red: array<f32, 256>;
fn wsum(tid: u32, v: f32) -> f32 {
  red[tid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

// ---------------------------------------------------------------- gemv
/**
 * y[r] = epi(act(W[r]·pro(x) + bias[r])), 8 rows per 256-thread workgroup. W is packed f16.
 * pro: 0 = x, 1 = LayerNorm(x) (affine when lnW >= 0), 2 = LN then *(1+scale)+shift.
 * epi: 0 = store, 1 = y += v, 2 = y += gate*v, 3 = y = res + v.
 */
export const GEMV: KernelDef = {
  name: 'gemv',
  fields: [
    ['m', 'u32'], ['n', 'u32'], ['xOff', 'u32'], ['xFS', 'u32'], ['yOff', 'u32'], ['yFS', 'u32'],
    ['pro', 'u32'], ['lnW', 'i32'], ['lnB', 'i32'], ['eps', 'f32'], ['shOff', 'u32'], ['scOff', 'u32'],
    ['bias', 'i32'], ['actf', 'u32'], ['epi', 'u32'], ['gOff', 'u32'], ['rOff', 'u32'], ['rFS', 'u32'],
  ],
  bindings: ['read_u32', 'rw', 'read'],
  body: /* wgsl */ `
${WSUM}
var<workgroup> xs: array<f32, 2048>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let frame = u32(act[0]);
  let n = p.n;
  let xo = p.xOff + frame * p.xFS;
  for (var j = tid; j < n; j = j + 256u) { xs[j] = act[xo + j]; }
  workgroupBarrier();
  if (p.pro != 0u) {
    var s = 0.0;
    for (var j = tid; j < n; j = j + 256u) { s = s + xs[j]; }
    let mu = wsum(tid, s) / f32(n);
    var q = 0.0;
    for (var j = tid; j < n; j = j + 256u) { let d = xs[j] - mu; q = q + d * d; }
    let den = sqrt(wsum(tid, q) / f32(n) + p.eps);
    for (var j = tid; j < n; j = j + 256u) {
      var y = (xs[j] - mu) / den;
      if (p.lnW >= 0) { y = y * W2[u32(p.lnW) + j] + W2[u32(p.lnB) + j]; }
      if (p.pro == 2u) { y = y * (1.0 + act[p.scOff + j]) + act[p.shOff + j]; }
      xs[j] = y;
    }
    workgroupBarrier();
  }
  let lane = tid & 31u;
  let row = wg.x * 8u + (tid >> 5u);
  var acc = 0.0;
  if (row < p.m) {
    let base = row * n;
    for (var j = lane * 4u; j < n; j = j + 128u) {
      let wi = (base + j) >> 1u; // base + j is a multiple of 4
      let w = vec4<f32>(unpack2x16float(W[wi]), unpack2x16float(W[wi + 1u]));
      acc = acc + dot(w, vec4<f32>(xs[j], xs[j + 1u], xs[j + 2u], xs[j + 3u]));
    }
  }
  red[tid] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s = s >> 1u) {
    if (lane < s) { red[tid] = red[tid] + red[tid + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < p.m) {
    var v = red[tid];
    if (p.bias >= 0) { v = v + W2[u32(p.bias) + row]; }
    v = act_f(v, p.actf);
    let yo = p.yOff + frame * p.yFS + row;
    if (p.epi == 0u) { act[yo] = v; }
    else if (p.epi == 1u) { act[yo] = act[yo] + v; }
    else if (p.epi == 2u) { act[yo] = act[yo] + act[p.gOff + row] * v; }
    else { act[yo] = act[p.rOff + frame * p.rFS + row] + v; }
  }
}
`,
};

// ---------------------------------------------------------------- gemm
/**
 * C[M][N] = epi(A[M][K] · W[N][K]^T). A row i starts at aOff + i*lda (lda may
 * be smaller than K: implicit im2col for causal convs over a history buffer).
 * epi: v = act(acc + bias[n]) * scale[n] (+ res[i][n]).
 */
export const GEMM: KernelDef = {
  name: 'gemm',
  fields: [
    ['M', 'u32'], ['N', 'u32'], ['K', 'u32'], ['aOff', 'u32'], ['lda', 'u32'], ['cOff', 'u32'], ['ldc', 'u32'],
    ['bias', 'i32'], ['actf', 'u32'], ['eluIn', 'u32'], ['scl', 'i32'], ['res', 'i32'],
  ],
  bindings: ['read_u32', 'rw', 'read'],
  body: /* wgsl */ `
fn w_at(i: u32) -> f32 {
  let v = unpack2x16float(W[i >> 1u]);
  return select(v.x, v.y, (i & 1u) == 1u);
}
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;
  let m0 = wg.y * 64u;
  let n0 = wg.x * 64u;
  var acc: array<vec4<f32>, 4>;
  for (var k0 = 0u; k0 < p.K; k0 = k0 + 16u) {
    for (var q = 0u; q < 4u; q = q + 1u) {
      let idx = tid + q * 256u;
      let r = idx >> 4u;
      let kk = idx & 15u;
      let gk = k0 + kk;
      var a = 0.0;
      let gm = m0 + r;
      if (gm < p.M && gk < p.K) {
        a = act[p.aOff + gm * p.lda + gk];
        if (p.eluIn != 0u) { a = elu_f(a); }
      }
      As[kk * 64u + r] = a;
      var b = 0.0;
      let gn = n0 + r;
      if (gn < p.N && gk < p.K) { b = w_at(gn * p.K + gk); }
      Bs[kk * 64u + r] = b;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let ab = kk * 64u + lid.y * 4u;
      let bb = kk * 64u + lid.x * 4u;
      let bv = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      acc[0] = acc[0] + As[ab] * bv;
      acc[1] = acc[1] + As[ab + 1u] * bv;
      acc[2] = acc[2] + As[ab + 2u] * bv;
      acc[3] = acc[3] + As[ab + 3u] * bv;
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i = i + 1u) {
    let row = m0 + lid.y * 4u + i;
    if (row >= p.M) { break; }
    for (var j = 0u; j < 4u; j = j + 1u) {
      let col = n0 + lid.x * 4u + j;
      if (col >= p.N) { break; }
      var v = acc[i][j];
      if (p.bias >= 0) { v = v + W2[u32(p.bias) + col]; }
      v = act_f(v, p.actf);
      if (p.scl >= 0) { v = v * W2[u32(p.scl) + col]; }
      let o = row * p.ldc + col;
      if (p.res >= 0) { v = v + act[u32(p.res) + o]; }
      act[p.cOff + o] = v;
    }
  }
}
`,
};

// ---------------------------------------------------------------- layernorm (rows)
export const LAYERNORM: KernelDef = {
  name: 'layernorm',
  fields: [['n', 'u32'], ['xOff', 'u32'], ['xS', 'u32'], ['yOff', 'u32'], ['yS', 'u32'], ['w', 'u32'], ['b', 'u32'], ['eps', 'f32']],
  bindings: ['rw', 'read'],
  body: /* wgsl */ `
${WSUM}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let xo = p.xOff + wg.x * p.xS;
  let yo = p.yOff + wg.x * p.yS;
  var s = 0.0;
  for (var j = tid; j < p.n; j = j + 256u) { s = s + act[xo + j]; }
  let mu = wsum(tid, s) / f32(p.n);
  var q = 0.0;
  for (var j = tid; j < p.n; j = j + 256u) { let d = act[xo + j] - mu; q = q + d * d; }
  let den = sqrt(wsum(tid, q) / f32(p.n) + p.eps);
  for (var j = tid; j < p.n; j = j + 256u) {
    act[yo + j] = (act[xo + j] - mu) / den * C[p.w + j] + C[p.b + j];
  }
}
`,
};

// ---------------------------------------------------------------- rope + kv write
/**
 * Rotary embedding on q (in place) and k, then k and v written to the f16 KV
 * cache slot pos % R (kOff/vOff/slots count halves). Each thread handles two
 * adjacent dims so every packed u32 has a single writer.
 * pos = posB + posFS*step + t; the rope table has [cos(32) | sin(32)] per position.
 */
export const ROPE_KV: KernelDef = {
  name: 'rope_kv',
  fields: [
    ['T', 'u32'], ['H', 'u32'], ['qOff', 'u32'], ['qS', 'u32'], ['posB', 'u32'], ['posFS', 'u32'],
    ['kOff', 'u32'], ['vOff', 'u32'], ['R', 'u32'], ['dyn', 'u32'],
  ],
  bindings: ['rw', 'read', 'rw_u32'],
  body: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= p.T * p.H * 16u) { return; }
  let i = (g & 15u) * 2u;
  let h = (g >> 4u) % p.H;
  let t = (g >> 4u) / p.H;
  let frame = u32(act[0]);
  var pos = p.posB + p.posFS * frame + t;
  if (p.dyn != 0u) { pos = pos + u32(act[1]); }
  let rb = pos * 64u + i;
  let c = vec2<f32>(rope[rb], rope[rb + 1u]);
  let s = vec2<f32>(rope[rb + 32u], rope[rb + 33u]);
  let hd = p.H * 64u;
  let qb = p.qOff + t * p.qS + h * 64u + i;
  let q1 = vec2<f32>(act[qb], act[qb + 1u]);
  let q2 = vec2<f32>(act[qb + 32u], act[qb + 33u]);
  let r1 = q1 * c - q2 * s;
  let r2 = q2 * c + q1 * s;
  act[qb] = r1.x; act[qb + 1u] = r1.y;
  act[qb + 32u] = r2.x; act[qb + 33u] = r2.y;
  let kb = qb + hd;
  let k1 = vec2<f32>(act[kb], act[kb + 1u]);
  let k2 = vec2<f32>(act[kb + 32u], act[kb + 33u]);
  let slot = (pos % p.R) * hd + h * 64u + i;
  kv[(p.kOff + slot) >> 1u] = pack2x16float(k1 * c - k2 * s);
  kv[(p.kOff + slot + 32u) >> 1u] = pack2x16float(k2 * c + k1 * s);
  let vb = kb + hd;
  kv[(p.vOff + slot) >> 1u] = pack2x16float(vec2<f32>(act[vb], act[vb + 1u]));
  kv[(p.vOff + slot + 32u) >> 1u] = pack2x16float(vec2<f32>(act[vb + 32u], act[vb + 33u]));
}
`,
};

// ---------------------------------------------------------------- attention
/** One 64-thread workgroup per (row, head); causal, optional sliding window, online softmax. K/V are f16 in `kv`. */
export const ATTENTION: KernelDef = {
  name: 'attention',
  fields: [
    ['H', 'u32'], ['qOff', 'u32'], ['qS', 'u32'], ['posB', 'u32'], ['posFS', 'u32'], ['kOff', 'u32'],
    ['vOff', 'u32'], ['R', 'u32'], ['win', 'u32'], ['oOff', 'u32'], ['oS', 'u32'], ['scale', 'f32'], ['dyn', 'u32'],
  ],
  bindings: ['rw', 'read_u32'],
  body: /* wgsl */ `
var<workgroup> qs: array<f32, 64>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> rd: array<f32, 64>;
var<workgroup> wfr: u32;
var<workgroup> wbase: u32;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) d: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  if (d == 0u) { wfr = u32(act[0]); wbase = select(0u, u32(act[1]), p.dyn != 0u); }
  let frame = workgroupUniformLoad(&wfr);
  let pos = p.posB + p.posFS * frame + t + workgroupUniformLoad(&wbase);
  var lo = 0u;
  if (p.win > 0u && pos + 1u > p.win) { lo = pos + 1u - p.win; }
  let hd = p.H * 64u;
  qs[d] = act[p.qOff + t * p.qS + h * 64u + d];
  workgroupBarrier();
  var m = -3.0e38;
  var l = 0.0;
  var acc = 0.0;
  for (var base = lo; base <= pos; base = base + 256u) {
    var lmax = -3.0e38;
    for (var u = 0u; u < 4u; u = u + 1u) {
      let jj = d + u * 64u;
      let j = base + jj;
      var s = -3.0e38;
      if (j <= pos) {
        let kw = (p.kOff + (j % p.R) * hd + h * 64u) >> 1u;
        var dt = 0.0;
        for (var e = 0u; e < 64u; e = e + 4u) {
          let kk = vec4<f32>(unpack2x16float(kv[kw + (e >> 1u)]), unpack2x16float(kv[kw + (e >> 1u) + 1u]));
          dt = dt + dot(vec4<f32>(qs[e], qs[e + 1u], qs[e + 2u], qs[e + 3u]), kk);
        }
        s = dt * p.scale;
      }
      sc[jj] = s;
      lmax = max(lmax, s);
    }
    rd[d] = lmax;
    workgroupBarrier();
    for (var s2 = 32u; s2 > 0u; s2 = s2 >> 1u) {
      if (d < s2) { rd[d] = max(rd[d], rd[d + s2]); }
      workgroupBarrier();
    }
    let mn = max(m, rd[0]);
    workgroupBarrier();
    let f = exp(m - mn);
    var ls = 0.0;
    for (var u = 0u; u < 4u; u = u + 1u) {
      let jj = d + u * 64u;
      let e = exp(sc[jj] - mn);
      sc[jj] = e;
      ls = ls + e;
    }
    rd[d] = ls;
    workgroupBarrier();
    for (var s2 = 32u; s2 > 0u; s2 = s2 >> 1u) {
      if (d < s2) { rd[d] = rd[d] + rd[d + s2]; }
      workgroupBarrier();
    }
    l = l * f + rd[0];
    acc = acc * f;
    let cnt = min(256u, pos + 1u - base);
    for (var jj = 0u; jj < cnt; jj = jj + 1u) {
      let vv = unpack2x16float(kv[(p.vOff + ((base + jj) % p.R) * hd + h * 64u + d) >> 1u]);
      acc = acc + sc[jj] * select(vv.x, vv.y, (d & 1u) == 1u);
    }
    m = mn;
    workgroupBarrier();
  }
  act[p.oOff + t * p.oS + h * 64u + d] = acc / l;
}
`,
};

// ---------------------------------------------------------------- lm final norm + eos head + step advance
export const LM_FINAL: KernelDef = {
  name: 'lm_final',
  fields: [
    ['xOff', 'u32'], ['hOff', 'u32'], ['w', 'u32'], ['b', 'u32'], ['eps', 'f32'], ['ew', 'u32'], ['eb', 'u32'],
    ['eosOff', 'u32'], ['add', 'u32'], ['inc', 'u32'],
  ],
  bindings: ['rw', 'read'],
  body: /* wgsl */ `
${WSUM}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32) {
  let frame = u32(act[0]);
  var s = 0.0;
  for (var j = tid; j < 512u; j = j + 256u) { s = s + act[p.xOff + j]; }
  let mu = wsum(tid, s) / 512.0;
  var q = 0.0;
  for (var j = tid; j < 512u; j = j + 256u) { let d = act[p.xOff + j] - mu; q = q + d * d; }
  let den = sqrt(wsum(tid, q) / 512.0 + p.eps);
  var e = 0.0;
  for (var j = tid; j < 512u; j = j + 256u) {
    let y = (act[p.xOff + j] - mu) / den * C[p.w + j] + C[p.b + j];
    act[p.hOff + j] = y;
    e = e + y * C[p.ew + j];
  }
  let eos = wsum(tid, e);
  if (tid == 0u) {
    act[p.eosOff + frame + p.add] = eos + C[p.eb];
    if (p.inc != 0u) { act[0] = f32(frame + 1u); }
  }
}
`,
};

// ---------------------------------------------------------------- depthwise transposed-conv upsample (stride 16, K 32)
export const UPSAMPLE: KernelDef = {
  name: 'upsample',
  fields: [['F', 'u32'], ['xOff', 'u32'], ['w', 'u32'], ['oOff', 'u32']],
  bindings: ['rw', 'read'],
  body: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= p.F * 16u * 512u) { return; }
  let c = g & 511u;
  let u = g >> 9u;
  let t = u >> 4u;
  let k = u & 15u;
  act[p.oOff + g] = act[p.xOff + (t + 1u) * 512u + c] * C[p.w + c * 32u + k]
                  + act[p.xOff + t * 512u + c] * C[p.w + c * 32u + k + 16u];
}
`,
};

// ---------------------------------------------------------------- copy (non-overlapping)
export const COPY: KernelDef = {
  name: 'copy',
  fields: [['n', 'u32'], ['src', 'u32'], ['dst', 'u32']],
  bindings: ['rw'],
  body: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < p.n) { act[p.dst + gid.x] = act[p.src + gid.x]; }
}
`,
};

// ---------------------------------------------------------------- f32 -> packed f16
/** packed[dst + i] = pack(stage[src + 2i], stage[src + 2i + 1]) for i < n (grid-stride). */
export const PACK_F16: KernelDef = {
  name: 'pack_f16',
  fields: [['n', 'u32'], ['src', 'u32'], ['dst', 'u32']],
  bindings: ['read', 'rw_u32'],
  body: /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var i = gid.x; i < p.n; i = i + nw.x * 256u) {
    packed[p.dst + i] = pack2x16float(vec2<f32>(stage[p.src + 2u * i], stage[p.src + 2u * i + 1u]));
  }
}
`,
};

// ---------------------------------------------------------------- trellis decode
const TRELLIS_FIELDS: Fields = [
  ['m', 'u32'], ['n', 'u32'], ['step', 'u32'], ['ws', 'f32'], ['codes', 'u32'], ['su', 'u32'], ['sv', 'u32'],
  ['mode', 'u32'], ['base', 'u32'], ['rs', 'u32'], ['cs', 'u32'], ['K', 'u32'], ['s', 'u32'], ['co', 'u32'],
  ['cin', 'u32'], ['tl', 'u32'],
];

const FWHT = /* wgsl */ `
var<workgroup> buf: array<f32, 2048>;
fn byte_at(i: u32) -> u32 { return (codes[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
fn fwht(tid: u32, len: u32) {
  for (var h = 1u; h < len; h = h << 1u) {
    for (var q = tid; q < (len >> 1u); q = q + 256u) {
      let i0 = (q / h) * 2u * h + (q % h);
      let a = buf[i0];
      let b = buf[i0 + h];
      buf[i0] = a + b;
      buf[i0 + h] = a - b;
    }
    workgroupBarrier();
  }
}
`;

/** Row pass: decode codes (Q*ws) for row i, rotate by the su-signed Hadamard. */
export const TRELLIS_ROWS: KernelDef = {
  name: 'trellis_rows',
  fields: TRELLIS_FIELDS,
  bindings: ['read_u32', 'read', 'rw'],
  body: /* wgsl */ `
${FWHT}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let i = wg.x;
  let n = p.n;
  let nb = n / 16u;
  let TB = 16u * p.step;
  let t = i / 16u;
  let r = i % 16u;
  for (var j = tid; j < n; j = j + 256u) {
    let jb = j / 16u;
    let k = (j % 16u) / 2u;
    let tile = p.codes + (t * nb + jb) * TB;
    let bit = (r * 8u + k) * p.step;
    let b0 = bit / 8u;
    let sh = bit % 8u;
    let v = (byte_at(tile + b0 % TB) << 16u) | (byte_at(tile + (b0 + 1u) % TB) << 8u) | byte_at(tile + (b0 + 2u) % TB);
    let s = (v >> (8u - sh)) & 65535u;
    let e = ((s * (s + 1u)) & 65535u) >> 6u;
    buf[j] = tlut[p.tl + 2u * e + (j & 1u)] * p.ws;
  }
  workgroupBarrier();
  fwht(tid, n);
  let sc = 1.0 / sqrt(f32(n));
  for (var j = tid; j < n; j = j + 256u) {
    let bitv = (byte_at(p.su + (j >> 3u)) >> (j & 7u)) & 1u;
    let v = buf[j] * sc;
    scratch[i * n + j] = select(-v, v, bitv == 1u);
  }
}
`,
};

/** Column pass: rotate column j by the sv-signed Hadamard and scatter into the dense weight. */
export const TRELLIS_COLS: KernelDef = {
  name: 'trellis_cols',
  fields: TRELLIS_FIELDS,
  bindings: ['read_u32', 'rw', 'rw'],
  body: /* wgsl */ `
${FWHT}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let j = wg.x;
  let m = p.m;
  for (var i = tid; i < m; i = i + 256u) { buf[i] = scratch[i * p.n + j]; }
  workgroupBarrier();
  fwht(tid, m);
  let sc = 1.0 / sqrt(f32(m));
  for (var i = tid; i < m; i = i + 256u) {
    let bitv = (byte_at(p.sv + (i >> 3u)) >> (i & 7u)) & 1u;
    var v = buf[i] * sc;
    v = select(-v, v, bitv == 1u);
    if (p.mode == 0u) {
      dest[p.base + i * p.rs + j * p.cs] = v;
    } else {
      let o = j / p.K;
      let kk = j % p.K;
      let row = (kk % p.s) * p.co + o;
      let col = select(0u, p.cin, kk < p.s) + i;
      dest[p.base + row * (2u * p.cin) + col] = v;
    }
  }
}
`,
};

/** Binding variable names per kernel (after the uniform). */
export const BINDING_NAMES: Record<string, readonly string[]> = {
  gemv: ['W', 'act', 'W2'],
  gemm: ['W', 'act', 'W2'],
  layernorm: ['act', 'C'],
  rope_kv: ['act', 'rope', 'kv'],
  attention: ['act', 'kv'],
  lm_final: ['act', 'C'],
  upsample: ['act', 'C'],
  copy: ['act'],
  trellis_rows: ['codes', 'tlut', 'scratch'],
  trellis_cols: ['codes', 'scratch', 'dest'],
  pack_f16: ['stage', 'packed'],
};

export const KERNELS = [GEMV, GEMM, LAYERNORM, ROPE_KV, ATTENTION, LM_FINAL, UPSAMPLE, COPY, TRELLIS_ROWS, TRELLIS_COLS, PACK_F16];
