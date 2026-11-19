import { InvalidPdfError, UnsupportedPdfError } from './errors';

export type PdfName = { name: string };
export type PdfRef = { ref: number };
export type PdfValue = null | boolean | number | string | PdfName | PdfRef | PdfValue[] | PdfDict;
export interface PdfDict { [key: string]: PdfValue; }

const isRef = (value: PdfValue | undefined): value is PdfRef => !!value && typeof value === 'object' && !Array.isArray(value) && 'ref' in value;
const isDict = (value: PdfValue | undefined): value is PdfDict => !!value && typeof value === 'object' && !Array.isArray(value) && !('ref' in value) && !('name' in value);

const latin1 = (bytes: Uint8Array) => {
  const chunks: string[] = [];
  for (let i = 0; i < bytes.length; i += 8192) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
  return chunks.join('');
};
const bytesFromLatin1 = (value: string): Uint8Array<ArrayBuffer> => Uint8Array.from(value, character => character.charCodeAt(0) & 0xff);
const latin1FromBytes = (bytes: Uint8Array) => {
  const chunks: string[] = [];
  for (let i = 0; i < bytes.length; i += 8192) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
  return chunks.join('');
};
const ws = /\s/;
/**
 * Force a flat, independent copy of a (possibly sliced) string. V8 keeps
 * substrings as views that pin their parent, so without this every token
 * would keep the whole-file string alive.
 */
const flat = (value: string) => (' ' + value).slice(1);

class Tokens {
  constructor(public readonly source: string, public pos = 0) {}

  skip() { while (this.pos < this.source.length && ws.test(this.source[this.pos]!)) this.pos++; }

  next(): string | null {
    this.skip();
    if (this.pos >= this.source.length) return null;
    const start = this.pos;
    const c = this.source[this.pos]!;
    if (c === '/') {
      this.pos++;
      while (this.pos < this.source.length && !ws.test(this.source[this.pos]!) && !'[]<>/()'.includes(this.source[this.pos]!)) this.pos++;
      return this.source.slice(start, this.pos);
    }
    if (c === '<' && this.source[this.pos + 1] !== '<') {
      this.pos++;
      while (this.pos < this.source.length && this.source[this.pos] !== '>') this.pos++;
      if (this.pos < this.source.length) this.pos++;
      return this.source.slice(start, this.pos);
    }
    if ('[]<>'.includes(c)) {
      this.pos += c === '<' && this.source[this.pos + 1] === '<' ? 2 : c === '>' && this.source[this.pos + 1] === '>' ? 2 : 1;
      return this.source.slice(start, this.pos);
    }
    if (c === '(') {
      this.pos++;
      let depth = 1;
      while (this.pos < this.source.length && depth) {
        if (this.source[this.pos] === '\\') this.pos += 2;
        else { if (this.source[this.pos] === '(') depth++; if (this.source[this.pos] === ')') depth--; this.pos++; }
      }
      return this.source.slice(start, this.pos);
    }
    while (this.pos < this.source.length && !ws.test(this.source[this.pos]!) && !'[]<>/'.includes(this.source[this.pos]!)) this.pos++;
    return this.source.slice(start, this.pos);
  }
}

function parseValue(t: Tokens): PdfValue {
  const token = t.next();
  if (token === null) throw new InvalidPdfError('Unexpected end of PDF object');
  if (token === '<<') {
    const dict: PdfDict = {};
    while (true) {
      const key = t.next();
      if (key === '>>') return dict;
      if (!key?.startsWith('/')) throw new InvalidPdfError('Invalid dictionary key');
      dict[key.slice(1)] = parseValue(t);
    }
  }
  if (token === '[') {
    const values: PdfValue[] = [];
    while (true) { if (t.source.slice(t.pos).trimStart().startsWith(']')) { t.next(); return values; } values.push(parseValue(t)); }
  }
  if (token.startsWith('/')) return { name: token.slice(1) };
  if (token.startsWith('(')) return decodeString(token.slice(1, -1));
  if (token.startsWith('<') && token.endsWith('>')) return token;
  if (token === 'true' || token === 'false') return token === 'true';
  if (token === 'null') return null;
  const number = Number(token);
  if (Number.isFinite(number)) {
    const save = t.pos;
    const generation = t.next();
    const marker = t.next();
    if (generation && marker === 'R' && /^\d+$/.test(generation)) return { ref: number };
    t.pos = save;
    return number;
  }
  throw new InvalidPdfError(`Unsupported PDF token: ${token}`);
}

function decodeString(value: string): string {
  let out = '';
  for (let i = 0; i < value.length; i++) {
    if (value[i] !== '\\') { out += value[i]; continue; }
    const next = value[++i] ?? '';
    const escapes: Record<string, string> = { n: '\n', r: '\r', t: '\t', b: '\b', f: '\f', '(': '(', ')': ')', '\\': '\\' };
    if (escapes[next]) out += escapes[next];
    else if (/[0-7]/.test(next)) { let oct = next; while (oct.length < 3 && /[0-7]/.test(value[i + 1] || '')) oct += value[++i]!; out += String.fromCharCode(parseInt(oct, 8)); }
    else out += next || '';
  }
  return out;
}

export class PdfFile {
  private readonly objects = new Map<number, PdfValue>();

  private constructor(bytes: Uint8Array) {
    // The whole-file latin1 string is only needed while scanning; objects keep
    // flat copies of their own text and streams keep copied byte ranges, so
    // neither this string nor the caller's ArrayBuffer is retained.
    const source = latin1(bytes);
    if (!source.startsWith('%PDF-')) throw new InvalidPdfError('Missing PDF header');
    this.readObjects(source, bytes);
  }

  static from(bytes: ArrayBuffer | Uint8Array) { return new PdfFile(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)); }

  get(id: number): PdfValue | undefined { return this.objects.get(id); }
  values() { return this.objects.values(); }

  resolve(value: PdfValue | undefined): PdfValue | undefined {
    let current = value;
    for (let i = 0; i < 16 && isRef(current); i++) current = this.objects.get(current.ref);
    return current;
  }

  private readObjects(source: string, bytes: Uint8Array) {
    const objectRe = /(\d+)\s+(\d+)\s+obj\b/g;
    let match: RegExpExecArray | null;
    while ((match = objectRe.exec(source))) {
      const id = Number(match[1]);
      const bodyStart = objectRe.lastIndex;
      const end = source.indexOf('endobj', bodyStart);
      if (end < 0) throw new InvalidPdfError(`Object ${id} has no endobj`);
      const body = flat(source.slice(bodyStart, end).trim());
      const streamAt = body.indexOf('stream');
      const parsed = streamAt >= 0 ? body.slice(0, streamAt).trim() : body;
      const value = parseValue(new Tokens(parsed));
      if (streamAt >= 0 && isDict(value) && 'Length' in value) {
        const streamKeyword = source.indexOf('stream', bodyStart);
        const afterStream = streamKeyword + 6;
        let start = afterStream;
        if (source[start] === '\r') start++;
        if (source[start] === '\n') start++;
        const lengthValue = this.resolve(value.Length);
        let length: number;
        if (typeof lengthValue === 'number') length = lengthValue;
        else {
          const endstream = source.indexOf('endstream', start);
          length = endstream - start;
          (value as PdfDict).__streamStart = start;
          if (value.Length && typeof value.Length === 'object' && 'ref' in value.Length) (value as PdfDict).__streamLengthRef = value.Length.ref;
        }
        (value as any).__stream = bytes.slice(start, start + length);
      }
      this.objects.set(id, value);
      objectRe.lastIndex = end + 6;
    }
    for (const value of this.objects.values()) {
      if (!isDict(value)) continue;
      const stream = value as PdfDict & { __stream?: Uint8Array; __streamStart?: number; __streamLengthRef?: number };
      if (typeof stream.__streamStart !== 'number' || typeof stream.__streamLengthRef !== 'number') continue;
      const length = this.resolve({ ref: stream.__streamLengthRef });
      if (typeof length === 'number') (stream as any).__stream = bytes.slice(stream.__streamStart, stream.__streamStart + length);
      delete stream.__streamStart; delete stream.__streamLengthRef;
    }
    if (!this.objects.size) throw new InvalidPdfError('No PDF objects found');
  }

  static pdfName(value: PdfValue | undefined): string | undefined { return value && typeof value === 'object' && !Array.isArray(value) && 'name' in value ? String(value.name) : undefined; }

  async stream(value: PdfValue | undefined): Promise<string> {
    return latin1FromBytes(await this.streamBytes(value));
  }

  async streamBytes(value: PdfValue | undefined): Promise<Uint8Array> {
    const resolved = this.resolve(value);
    if (!isDict(resolved) || !('__stream' in resolved)) throw new InvalidPdfError('Expected PDF stream');
    // Filter may be a single name or an array applied in order (e.g. [/FlateDecode]).
    const filterValue = this.resolve(resolved.Filter);
    const filters = (Array.isArray(filterValue) ? filterValue : [filterValue]).map(f => PdfFile.pdfName(this.resolve(f))).filter(Boolean) as string[];
    const parmsValue = this.resolve(resolved.DecodeParms);
    const parmsList = Array.isArray(parmsValue) ? parmsValue : [parmsValue];
    const raw = (resolved as any).__stream;
    let bytes: Uint8Array<ArrayBuffer> = raw instanceof Uint8Array ? raw as Uint8Array<ArrayBuffer> : bytesFromLatin1(String(raw));
    for (let i = 0; i < filters.length; i++) {
      const filter = filters[i]!;
      if (filter === 'None') continue;
      if (filter === 'DCTDecode') return bytes;
      if (filter !== 'FlateDecode') throw new UnsupportedPdfError(`Stream filter ${filter} is not implemented yet`);
      if (typeof DecompressionStream === 'undefined') throw new UnsupportedPdfError('FlateDecode requires DecompressionStream');
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
      bytes = new Uint8Array(await new Response(stream).arrayBuffer());
      const parms = this.resolve(parmsList[i] as PdfValue | undefined);
      if (isDict(parms)) bytes = applyPredictor(bytes, parms);
    }
    return bytes;
  }

  /**
   * PDF 1.5+ files usually keep most objects (often the catalog, page tree and
   * pages) inside compressed /ObjStm object streams, which the textual
   * `N G obj` scan cannot see. Unpack them into the object table. Objects
   * written directly in the file win, since they are typically newer
   * incremental updates.
   */
  async expandObjectStreams() {
    const streams = [...this.objects.values()].filter((v): v is PdfDict => isDict(v) && PdfFile.pdfName((v as PdfDict).Type) === 'ObjStm' && '__stream' in v);
    for (const objStm of streams) {
      let text: string;
      try { text = await this.stream(objStm); } catch { continue; }
      const count = this.resolve(objStm.N), first = this.resolve(objStm.First);
      if (typeof count !== 'number' || typeof first !== 'number') continue;
      const header = text.slice(0, first).trim().split(/\s+/).map(Number);
      const entries: Array<[number, number]> = [];
      for (let i = 0; i < count && 2 * i + 1 < header.length; i++) entries.push([header[2 * i]!, header[2 * i + 1]!]);
      entries.forEach(([id, offset], i) => {
        if (this.objects.has(id) || !Number.isFinite(id) || !Number.isFinite(offset)) return;
        const end = i + 1 < entries.length ? first + entries[i + 1]![1] : text.length;
        try { this.objects.set(id, parseValue(new Tokens(flat(text.slice(first + offset, end))))); } catch { /* skip malformed entry */ }
      });
    }
  }
}

/** Undo PNG (Predictor >= 10) or TIFF (2) row prediction after FlateDecode. */
function applyPredictor(data: Uint8Array<ArrayBuffer>, parms: PdfDict): Uint8Array<ArrayBuffer> {
  const predictor = typeof parms.Predictor === 'number' ? parms.Predictor : 1;
  if (predictor < 2) return data;
  const colors = typeof parms.Colors === 'number' ? parms.Colors : 1;
  const bpc = typeof parms.BitsPerComponent === 'number' ? parms.BitsPerComponent : 8;
  const columns = typeof parms.Columns === 'number' ? parms.Columns : 1;
  const bpp = Math.max(1, Math.ceil((colors * bpc) / 8));
  const rowLength = Math.ceil((colors * bpc * columns) / 8);
  if (predictor === 2) {
    if (bpc !== 8) return data;
    const out = data.slice();
    for (let r = 0; r + rowLength <= out.length; r += rowLength)
      for (let i = bpp; i < rowLength; i++) out[r + i] = (out[r + i]! + out[r + i - bpp]!) & 0xff;
    return out;
  }
  const rows = Math.floor(data.length / (rowLength + 1));
  const out = new Uint8Array(rows * rowLength);
  for (let r = 0; r < rows; r++) {
    const type = data[r * (rowLength + 1)]!;
    const src = r * (rowLength + 1) + 1, dst = r * rowLength, prev = dst - rowLength;
    for (let i = 0; i < rowLength; i++) {
      const raw = data[src + i]!;
      const left = i >= bpp ? out[dst + i - bpp]! : 0;
      const up = r > 0 ? out[prev + i]! : 0;
      const upLeft = r > 0 && i >= bpp ? out[prev + i - bpp]! : 0;
      let value: number;
      switch (type) {
        case 1: value = raw + left; break;
        case 2: value = raw + up; break;
        case 3: value = raw + ((left + up) >> 1); break;
        case 4: {
          const p = left + up - upLeft, pa = Math.abs(p - left), pb = Math.abs(p - up), pc = Math.abs(p - upLeft);
          value = raw + (pa <= pb && pa <= pc ? left : pb <= pc ? up : upLeft);
          break;
        }
        default: value = raw;
      }
      out[dst + i] = value & 0xff;
    }
  }
  return out;
}

export function parsePdf(bytes: ArrayBuffer | Uint8Array) { return PdfFile.from(bytes); }
