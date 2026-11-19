/**
 * Minimal port of the HuggingFace `tokenizers` pipeline used by Mach Boom:
 * a Sequence of Replace/Strip/Prepend normalizers, a Metaspace pre-tokenizer
 * and a Unigram model with byte fallback. Interprets tokenizer.json directly.
 */

type Normalizer = (s: string) => string;

/** Oniguruma (Ruby syntax) regex source -> JS regex source (with the `u` flag). */
function onigToJs(src: string): string {
  return src.replace(/\\x\{([0-9A-Fa-f]+)\}/g, '\\u{$1}');
}

function buildNormalizer(spec: any): Normalizer {
  if (!spec) return (s) => s;
  switch (spec.type) {
    case 'Sequence': {
      const parts = (spec.normalizers as any[]).map(buildNormalizer);
      return (s) => { for (const p of parts) s = p(s); return s; };
    }
    case 'Replace': {
      const content = spec.content as string;
      if (spec.pattern.String !== undefined) {
        const lit = spec.pattern.String as string;
        return (s) => s.split(lit).join(content);
      }
      const re = new RegExp(onigToJs(spec.pattern.Regex), 'gu');
      return (s) => s.replace(re, () => content);
    }
    case 'Strip': {
      const l = spec.strip_left !== false, r = spec.strip_right !== false;
      return (s) => {
        if (l) s = s.replace(/^\s+/u, '');
        if (r) s = s.replace(/\s+$/u, '');
        return s;
      };
    }
    case 'Prepend': {
      const p = spec.prepend as string;
      return (s) => (s.length ? p + s : s);
    }
    case 'NFC': case 'NFD': case 'NFKC': case 'NFKD':
      return (s) => s.normalize(spec.type);
    case 'Lowercase':
      return (s) => s.toLowerCase();
    default:
      console.warn(`[MachBoom tokenizer] unsupported normalizer ${spec.type}; ignoring`);
      return (s) => s;
  }
}

interface TrieNode { next: Map<string, TrieNode>; id: number }

export class UnigramTokenizer {
  private normalize: Normalizer;
  private replacement: string;
  private prependAlways: boolean;
  private root: TrieNode = { next: new Map(), id: -1 };
  private scores: Float64Array;
  private pieces: string[];
  private ids = new Map<string, number>();
  private unkId: number;
  private unkScore: number;
  private byteFallback: boolean;
  private added: { content: string; id: number; special: boolean }[];
  private specialIds = new Set<number>();

  constructor(json: any) {
    this.normalize = buildNormalizer(json.normalizer);
    const pt = json.pre_tokenizer ?? {};
    this.replacement = pt.replacement ?? '▁';
    this.prependAlways = (pt.prepend_scheme ?? 'always') === 'always';
    const m = json.model;
    if (m.type !== 'Unigram') throw new Error(`unsupported tokenizer model ${m.type}`);
    const vocab = m.vocab as [string, number][];
    this.scores = new Float64Array(vocab.length);
    this.pieces = vocab.map((v) => v[0]);
    let minScore = Infinity;
    vocab.forEach(([piece, score], id) => {
      this.scores[id] = score;
      if (!this.ids.has(piece)) this.ids.set(piece, id);
      if (score < minScore) minScore = score;
      let node = this.root;
      for (const ch of piece) {
        let nx = node.next.get(ch);
        if (!nx) { nx = { next: new Map(), id: -1 }; node.next.set(ch, nx); }
        node = nx;
      }
      if (node.id < 0) node.id = id;
    });
    this.unkId = m.unk_id ?? 0;
    this.unkScore = minScore - 10;
    this.byteFallback = !!m.byte_fallback;
    this.added = ((json.added_tokens ?? []) as any[])
      .map((a) => ({ content: a.content as string, id: a.id as number, special: !!a.special }))
      .sort((a, b) => b.content.length - a.content.length);
    for (const a of this.added) if (a.special) this.specialIds.add(a.id);
  }

  /** Viterbi segmentation of one pre-token; returns piece strings (unk runs fused). */
  private viterbi(chars: string[]): { str: string; unk: boolean }[] {
    const n = chars.length;
    const best = new Float64Array(n + 1);
    const from = new Int32Array(n + 1).fill(-1);
    const tok = new Int32Array(n + 1).fill(-1);
    from[0] = 0;
    for (let i = 0; i < n; i++) {
      if (from[i]! < 0 && i > 0) continue;
      const base = best[i]!;
      let node: TrieNode | undefined = this.root;
      let single = false;
      for (let j = i; j < n; j++) {
        node = node.next.get(chars[j]!);
        if (!node) break;
        if (node.id >= 0) {
          const end = j + 1;
          const cand = base + this.scores[node.id]!;
          if (from[end]! < 0 || cand > best[end]!) { best[end] = cand; from[end] = i; tok[end] = node.id; }
          if (end === i + 1) single = true;
        }
      }
      if (!single) {
        const end = i + 1;
        const cand = base + this.unkScore;
        if (from[end]! < 0 || cand > best[end]!) { best[end] = cand; from[end] = i; tok[end] = this.unkId; }
      }
    }
    const out: { str: string; unk: boolean }[] = [];
    for (let e = n; e > 0; e = from[e]!) {
      out.push({ str: chars.slice(from[e]!, e).join(''), unk: tok[e] === this.unkId });
    }
    out.reverse();
    const fused: { str: string; unk: boolean }[] = [];
    for (const p of out) {
      const last = fused[fused.length - 1];
      if (p.unk && last?.unk) last.str += p.str;
      else fused.push({ ...p });
    }
    return fused;
  }

  private encodeSegment(text: string, ids: number[]) {
    let s = this.normalize(text);
    if (!s) return;
    s = s.split(' ').join(this.replacement);
    if (this.prependAlways && !s.startsWith(this.replacement)) s = this.replacement + s;
    // Metaspace split, MergedWithNext: every replacement char starts a new pre-token.
    const words: string[] = [];
    let cur = '';
    for (const ch of s) {
      if (ch === this.replacement && cur) { words.push(cur); cur = ''; }
      cur += ch;
    }
    if (cur) words.push(cur);
    const enc = new TextEncoder();
    for (const w of words) {
      for (const p of this.viterbi(Array.from(w))) {
        const id = p.unk ? undefined : this.ids.get(p.str);
        if (id !== undefined) { ids.push(id); continue; }
        if (this.byteFallback) {
          const bytes = enc.encode(p.str);
          const bids: number[] = [];
          for (const b of bytes) {
            const bid = this.ids.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
            if (bid === undefined) { bids.length = 0; break; }
            bids.push(bid);
          }
          if (bids.length === bytes.length) { ids.push(...bids); continue; }
        }
        ids.push(this.unkId);
      }
    }
  }

  /** tokenizer.encode(text, add_special_tokens=False).ids */
  encode(text: string): number[] {
    const ids: number[] = [];
    let rest = text;
    while (rest) {
      let at = -1, hit: { content: string; id: number } | null = null;
      for (const a of this.added) {
        const k = rest.indexOf(a.content);
        if (k >= 0 && (at < 0 || k < at)) { at = k; hit = a; }
      }
      if (!hit) { this.encodeSegment(rest, ids); break; }
      if (at > 0) this.encodeSegment(rest.slice(0, at), ids);
      ids.push(hit.id);
      rest = rest.slice(at + hit.content.length);
    }
    return ids;
  }

  /** tokenizer.decode(ids, skip_special_tokens=True) */
  decode(ids: number[]): string {
    const bytes: number[] = [];
    let out = '';
    const flush = () => { if (bytes.length) { out += new TextDecoder().decode(new Uint8Array(bytes)); bytes.length = 0; } };
    for (const id of ids) {
      if (this.specialIds.has(id)) continue;
      const p = this.pieces[id] ?? '';
      const m = /^<0x([0-9A-F]{2})>$/.exec(p);
      if (m) { bytes.push(parseInt(m[1]!, 16)); continue; }
      flush();
      out += p;
    }
    flush();
    out = out.split(this.replacement).join(' ');
    return out.startsWith(' ') ? out.slice(1) : out;
  }

  /** MachBoom.text_ids: ids for the stripped text, plus its word count. */
  textIds(text: string): { ids: number[]; numWords: number } {
    const ids = this.encode(text.trim());
    const numWords = this.decode(ids).split(/\s+/u).filter(Boolean).length;
    return { ids, numWords };
  }
}
