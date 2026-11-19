/**
 * English text front end for Mach Boom: a port of frontend_en.py (numbers,
 * dates, times, money, units, codes and abbreviations as words).
 *
 * Python `re` is Unicode-aware for \w and \b while JS is ASCII-only, so
 * patterns are written in Python syntax and translated by `py()`.
 */

const W = '\\p{L}\\p{N}_';
const WORD_B = `(?:(?<=[${W}])(?![${W}])|(?<![${W}])(?=[${W}]))`;

/** Translate a Python regex to a JS RegExp with Unicode \w and \b semantics. */
function py(src: string, flags = 'g'): RegExp {
  let out = '';
  let inClass = false;
  for (let i = 0; i < src.length; i++) {
    const c = src[i]!;
    if (c === '\\') {
      const n = src[i + 1]!;
      i++;
      if (n === 'w') out += inClass ? W : `[${W}]`;
      else if (n === 'W') out += inClass ? '\\P{L}' : `[^${W}]`; // \W only occurs outside classes
      else if (n === 'b' && !inClass) out += WORD_B;
      else if (n === 'd') out += inClass ? '0-9' : '[0-9]';
      else if (n === 'D') out += '[^0-9]';
      else if (n === ' ') out += ' ';
      else out += '\\' + n;
      continue;
    }
    if (c === '[' && !inClass) inClass = true;
    else if (c === ']' && inClass) inClass = false;
    out += c;
  }
  return new RegExp(out, flags.includes('u') ? flags : flags + 'u');
}

interface M { group(i: number): string | null; readonly length: number }

type Repl = string | ((m: M) => string);

const reCache = new Map<string, RegExp>();
function rx(src: string, flags = 'g'): RegExp {
  const k = flags + '\u0000' + src;
  let r = reCache.get(k);
  if (!r) { r = py(src, flags); reCache.set(k, r); }
  r.lastIndex = 0;
  return r;
}

/** re.sub(pattern, repl, s) */
function sub(pattern: string, repl: Repl, s: string): string {
  const re = rx(pattern);
  if (typeof repl === 'string') {
    const r = repl.replace(/\\(\d)/g, (_, d) => `$${d}`);
    return s.replace(re, r);
  }
  return s.replace(re, (...a: any[]) => {
    let k = a.length - 1;
    if (typeof a[k] === 'object') k--; // named groups object
    const groups = a.slice(0, k - 1) as (string | undefined)[];
    return repl({ group: (i) => groups[i] ?? null, length: groups.length });
  });
}

function fullmatch(pattern: string, s: string): boolean {
  return rx(`^(?:${pattern})$`, '').test(s);
}
function search(pattern: string, s: string): boolean {
  return rx(pattern, '').test(s);
}
function findall(pattern: string, s: string): string[] {
  return s.match(rx(pattern)) ?? [];
}

const ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(' ');
const TENS = '_ _ twenty thirty forty fifty sixty seventy eighty ninety'.split(' ');
const ORD_IRREG: Record<string, string> = {
  one: 'first', two: 'second', three: 'third', five: 'fifth', eight: 'eighth', nine: 'ninth', twelve: 'twelfth',
};
const MONTHS = 'January February March April May June July August September October November December'.split(' ');
const FRAC: Record<string, string> = {
  '1/2': 'one half', '1/3': 'one third', '2/3': 'two thirds', '1/4': 'one quarter', '3/4': 'three quarters',
  '1/5': 'one fifth', '1/8': 'one eighth', '1/10': 'one tenth',
};

const isDigit = (c: string) => c >= '0' && c <= '9';
const isAlpha = (c: string) => /^\p{L}$/u.test(c);

function toInt(s: string | number | bigint): bigint {
  if (typeof s === 'bigint') return s;
  if (typeof s === 'number') return BigInt(Math.trunc(s));
  const t = s.trim().replace(/_/g, '');
  if (!/^[+-]?[0-9]+$/.test(t)) throw new Error(`invalid literal for int(): '${s}'`);
  return BigInt(t);
}

const SCALES: [bigint, string][] = [[10n ** 12n, 'trillion'], [10n ** 9n, 'billion'], [10n ** 6n, 'million'], [1000n, 'thousand']];

export function cardinal(v: string | number | bigint): string {
  const n = toInt(v);
  if (n < 0n) return 'minus ' + cardinal(-n);
  if (n < 20n) return ONES[Number(n)]!;
  if (n < 100n) return TENS[Number(n / 10n)]! + (n % 10n === 0n ? '' : '-' + ONES[Number(n % 10n)]);
  if (n < 1000n) return ONES[Number(n / 100n)]! + ' hundred' + (n % 100n === 0n ? '' : ' ' + cardinal(n % 100n));
  for (const [div, name] of SCALES) {
    if (n >= div) {
      const rest = n % div;
      return cardinal(n / div) + ' ' + name + (rest === 0n ? '' : ' ' + cardinal(rest));
    }
  }
  return String(n);
}

function rpartition(s: string, sep: string): [string, string, string] {
  const i = s.lastIndexOf(sep);
  return i < 0 ? ['', '', s] : [s.slice(0, i), sep, s.slice(i + sep.length)];
}

export function ordinal(n: string | number | bigint): string {
  const w = cardinal(n);
  const [head, sep, last0] = w.includes(' ') ? rpartition(w, ' ') : ['', '', w];
  let [hyHead, hy, tail] = last0.includes('-') ? rpartition(last0, '-') : ['', '', last0];
  if (tail in ORD_IRREG) tail = ORD_IRREG[tail]!;
  else if (tail.endsWith('y')) tail = tail.slice(0, -1) + 'ieth';
  else tail = tail + 'th';
  return head + sep + hyHead + hy + tail;
}

function year(v: string | number): string {
  const y = Number(toInt(v));
  if (y >= 2000 && y <= 2009) return 'two thousand' + (y === 2000 ? '' : ' ' + ONES[y - 2000]);
  if (y >= 1100 && y <= 2099 && y % 100 !== 0) {
    const hi = Math.floor(y / 100), lo = y % 100;
    return cardinal(hi) + ' ' + (lo < 10 ? 'oh ' + ONES[lo] : cardinal(lo));
  }
  if (y >= 1100 && y <= 2099) return cardinal(Math.floor(y / 100)) + ' hundred';
  return cardinal(y);
}

function digits(s: string): string {
  return Array.from(s).filter(isDigit).map((c) => ONES[+c]).join(' ');
}

function decimal(s0: string): string {
  let s = s0.replace(/,/g, '');
  const neg = s.startsWith('-') || s.startsWith('\u2212');
  s = s.replace(/^[-\u2212]+/, '');
  let out: string;
  if (s.includes('.')) {
    const k = s.indexOf('.');
    const a = s.slice(0, k), b = s.slice(k + 1);
    out = (a ? cardinal(a) : 'zero') + ' point ' + digits(b);
  } else out = cardinal(s);
  return (neg ? 'minus ' : '') + out;
}

function number(s: string): string {
  const raw = s.replace(/,/g, '');
  if (raw.includes('.')) return decimal(s);
  if ((raw.length >= 5 && !s.includes(',')) || (raw.length > 1 && raw.startsWith('0'))) return digits(raw);
  return cardinal(raw);
}

function spell(tok: string): string {
  const out: string[] = [];
  for (const c of tok) {
    if (isDigit(c)) out.push(ONES[+c]!);
    else if (isAlpha(c)) out.push(c.toUpperCase());
  }
  return out.join(' ');
}

const EXT_WORDS = new Set(['zip', 'html', 'json', 'yaml', 'java', 'py']);

function urlPiece(p: string): string {
  if (!p) return '';
  if (/^[0-9]+$/.test(p)) return digits(p);
  if (/^[A-Za-z]+$/.test(p)) return p;
  return findall('[A-Za-z]+|\\d+', p).map(urlPiece).join(' ');
}

function sayPath(s: string): string {
  const out: string[] = [];
  const names: Record<string, string> = { '.': 'dot', '/': 'slash', _: 'underscore', '-': 'dash', ':': 'colon', '@': 'at' };
  for (const part of s.split(/([./_\-:@])/)) {
    if (part in names) out.push(names[part]!);
    else if (part) out.push(urlPiece(part));
  }
  return out.filter(Boolean).join(' ');
}

const rEmail = (m: M) => sayPath(m.group(1)!) + ' at ' + sayPath(m.group(2)!);

function rUrl(m: M): string {
  const s = m.group(0)!.replace(/^https?:\/\//, '');
  return sayPath(s.replace(/[./]+$/, ''));
}

function rFile(m: M): string {
  const name = m.group(1)!, ext = m.group(2)!;
  const e = EXT_WORDS.has(ext.toLowerCase()) ? ext.toLowerCase() : spell(ext);
  return sayPath(name) + ' dot ' + e;
}

function rVersion(m: M): string {
  const pre = m.group(1), ver = m.group(2)!;
  const lead = pre && pre.toLowerCase() === 'v' ? 'version ' : pre ? pre : '';
  return lead + ver.split('.').map((p) => cardinal(p)).join(' point ');
}

function rPhone(m: M): string {
  const s = m.group(0)!;
  const plus = s.trim().startsWith('+') ? 'plus ' : '';
  return plus + findall('\\d+', s).map(digits).join(', ');
}

function minutes(mm: string): string {
  const v = Number(toInt(mm));
  return v > 0 && v < 10 ? 'oh ' + ONES[v] : cardinal(v);
}

function rTime(m: M): string {
  const h = Number(toInt(m.group(1)!)), mm = m.group(2)!, ap = m.group(3);
  let apS = '';
  if (ap) apS = ap.toLowerCase().startsWith('a') ? ' a m' : ' p m';
  if (Number(toInt(mm)) === 0) {
    if (ap) return cardinal(h) + apS;
    if (h > 12 || h === 0) return (h ? cardinal(h) : 'zero') + ' hundred';
    return cardinal(h) + " o'clock";
  }
  let hour = m.group(1)!.startsWith('0') && !ap && h !== 0 ? 'oh ' + ONES[h] : cardinal(h);
  if (h === 0) hour = 'zero';
  return hour + ' ' + minutes(mm) + apS;
}

function rDuration(m: M): string {
  const h = Number(toInt(m.group(1)!)), mm = m.group(2);
  let s = cardinal(h) + (h === 1 ? ' hour' : ' hours');
  if (mm) s += ' ' + cardinal(mm) + (m.group(3) ? ' minutes' : '');
  return s;
}

function dateWords(d: number, mo: number, y: string | null = null, us = false): string {
  const month = MONTHS.at(mo - 1)!;
  let s = us ? month + ' ' + ordinal(d) : 'the ' + ordinal(d) + ' of ' + month;
  if (y !== null && y !== undefined) {
    let yy = Number(toInt(y));
    if (y.length === 2) yy += yy < 50 ? 2000 : 1900;
    s += ', ' + year(yy);
  }
  return s;
}

function rSlashDate(m: M): string {
  const a = Number(toInt(m.group(1)!)), b = Number(toInt(m.group(2)!)), y = m.group(3);
  if (a > 12 && b <= 12) return dateWords(a, b, y);
  if (b > 12 && a <= 12) return dateWords(b, a, y, true);
  if (a <= 12 && b <= 12 && a >= 1 && b >= 1) return dateWords(a, b, y);
  return m.group(0)!;
}

function rWrittenMd(month: string, day: string, y: string | null): string {
  let s = month + ' ' + ordinal(day);
  if (y) s += ', ' + year(y);
  return s;
}

function rWrittenDm(m: M): string {
  let s = 'the ' + ordinal(m.group(1)!) + ' of ' + m.group(2);
  if (m.group(3)) s += ' ' + year(m.group(3)!);
  return s;
}

const CUR: Record<string, [string, string, string, string]> = {
  $: ['dollar', 'dollars', 'cent', 'cents'], '€': ['euro', 'euros', 'cent', 'cents'],
  '£': ['pound', 'pounds', 'penny', 'pence'], '¥': ['yen', 'yen', '', ''],
};
const SCALE: Record<string, string> = { k: 'thousand', m: 'million', mn: 'million', b: 'billion', bn: 'billion' };

function rCurrency(sym: string, amt: string, scaleRaw: string | null): string {
  const scale = (scaleRaw ?? '').toLowerCase();
  const [one, many, c1, cn] = CUR[sym]!;
  if (scale) return decimal(amt) + ' ' + SCALE[scale] + ' ' + many;
  const raw = amt.replace(/,/g, '');
  const k = raw.indexOf('.');
  const whole = k < 0 ? raw : raw.slice(0, k), frac = k < 0 ? '' : raw.slice(k + 1);
  const w = toInt(whole || '0');
  let s = cardinal(w) + ' ' + (w === 1n ? one : many);
  if (frac && toInt(frac) !== 0n && c1) {
    const f = Number(toInt(frac.padEnd(2, '0').slice(0, 2)));
    s += ' and ' + cardinal(f) + ' ' + (f === 1 ? c1 : cn);
  }
  return s;
}

const rPercent = (m: M) => decimal(m.group(1)!) + ' percent';

function rTemp(m: M): string {
  const u = (m.group(2) ?? '').toUpperCase();
  const unit = u === 'C' ? ' degrees Celsius' : u === 'F' ? ' degrees Fahrenheit' : ' degrees';
  return decimal(m.group(1)!) + unit;
}

const yearish = (x: string) => fullmatch('1[5-9]\\d\\d|20\\d\\d', x);

const UNITS: Record<string, [string, string]> = {
  mm: ['millimeter', 'millimeters'], cm: ['centimeter', 'centimeters'], m: ['meter', 'meters'],
  km: ['kilometer', 'kilometers'], kg: ['kilogram', 'kilograms'], g: ['gram', 'grams'],
  mg: ['milligram', 'milligrams'], l: ['liter', 'liters'], ml: ['milliliter', 'milliliters'],
  lb: ['pound', 'pounds'], lbs: ['pound', 'pounds'], oz: ['ounce', 'ounces'],
  ft: ['foot', 'feet'], in: ['inch', 'inches'], mi: ['mile', 'miles'],
  'km/h': ['kilometer per hour', 'kilometers per hour'], kph: ['kilometer per hour', 'kilometers per hour'],
  mph: ['mile per hour', 'miles per hour'], kwh: ['kilowatt hour', 'kilowatt hours'],
  kw: ['kilowatt', 'kilowatts'], w: ['watt', 'watts'], v: ['volt', 'volts'], mah: ['milliamp hour', 'milliamp hours'],
  mAh: ['milliamp hour', 'milliamp hours'], ha: ['hectare', 'hectares'],
  gb: ['gigabyte', 'gigabytes'], mb: ['megabyte', 'megabytes'], tb: ['terabyte', 'terabytes'],
  kb: ['kilobyte', 'kilobytes'], ghz: ['gigahertz', 'gigahertz'], mhz: ['megahertz', 'megahertz'],
  hz: ['hertz', 'hertz'], bar: ['bar', 'bar'], psi: ['p s i', 'p s i'], ms: ['millisecond', 'milliseconds'],
  s: ['second', 'seconds'], sec: ['second', 'seconds'], min: ['minute', 'minutes'],
  mins: ['minute', 'minutes'], h: ['hour', 'hours'], hr: ['hour', 'hours'], hrs: ['hour', 'hours'],
  'sq ft': ['square foot', 'square feet'], m2: ['square meter', 'square meters'],
  'm²': ['square meter', 'square meters'], '°': ['degree', 'degrees'],
};
const UNIT_RE = Object.keys(UNITS)
  .map((u) => u.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&'))
  .sort((a, b) => b.length - a.length)
  .join('|');

function rRange(m: M): string {
  const a = m.group(1)!, b = m.group(2)!, unit = m.group(3);
  let s = yearish(a) && yearish(b) ? year(a) + ' to ' + year(b) : number(a) + ' to ' + number(b);
  if (unit) s += ' ' + UNITS[unit.toLowerCase()]![1];
  return s;
}

function rUnit(m: M): string {
  const num = m.group(1)!, unit = m.group(2)!;
  const [one, many] = UNITS[unit.toLowerCase()]!;
  const val = num.replace(/,/g, '');
  return decimal(num) + ' ' + (val === '1' || val === '1.0' ? one : many);
}

const ABBR: [string, string][] = [
  ['\\bMr\\.', 'Mister'], ['\\bMrs\\.', 'Missus'], ['\\bMs\\.', 'Miz'], ['\\bDr\\.(?=\\s+[A-Z])', 'Doctor'],
  ['\\bProf\\.', 'Professor'], ['\\bSt\\.(?=\\s+[A-Z])', 'Saint'], ['(?<=[a-z] )St\\.', 'Street'],
  ['(?<=[a-z] )St\\b\\.?', 'Street'], ['\\bAve\\.?(?=\\W|$)', 'Avenue'], ['\\bRd\\.(?=\\W|$)', 'Road'],
  ['\\bBlvd\\.?', 'Boulevard'], ['\\bApt\\.?(?=\\s)', 'apartment'], ['\\b[Nn]o\\.(?=\\s*\\d)', 'number'],
  ['\\betc\\.', 'et cetera'], ['\\be\\.g\\.', 'for example'], ['\\bi\\.e\\.', 'that is'], ['\\bvs\\.?(?=\\s)', 'versus'],
  ['\\bapprox\\.', 'approximately'], ['\\bJr\\.', 'Junior'], ['\\bSr\\.', 'Senior'], ['\\bDept\\.', 'department'],
  ['\\bLtd\\.?', 'Limited'], ['\\bInc\\.', 'Incorporated'], ['\\bCo\\.(?=\\s|$)', 'Company'],
  ['\\bmin\\.(?=\\s)', 'minutes'], ['\\bhrs?\\.(?=\\s)', 'hours'], ['\\bft\\.(?=\\s)', 'feet'],
  ['\\bTel\\.?(?=\\s*[:\\d])', 'telephone'], ['\\bext\\.(?=\\s*\\d)', 'extension'], ['\\bMt\\.(?=\\s+[A-Z])', 'Mount'],
  ['\\b(a)\\.m\\.', 'A M'], ['\\b(p)\\.m\\.', 'P M'],
  ['\\bref\\.\\s+no\\.', 'reference number'], ['\\bSgt\\.', 'Sergeant'], ['\\bCapt\\.', 'Captain'],
  ['\\bLt\\.', 'Lieutenant'], ['\\b[Ff]ig\\.(?=\\s*\\d)', 'figure'], ['\\bCtrl\\s?\\+\\s?', 'control plus '],
  ['\\bCmd\\s?\\+\\s?', 'command plus '], ['\\bAlt\\s?\\+\\s?', 'alt plus '], ['\\bShift\\s?\\+\\s?', 'shift plus '],
  ['\\bMon\\.', 'Monday'], ['\\bTue\\.', 'Tuesday'], ['\\bWed\\.', 'Wednesday'], ['\\bThu\\.', 'Thursday'],
  ['\\bFri\\.', 'Friday'], ['\\bSat\\.', 'Saturday'], ['\\bSun\\.', 'Sunday'], ['\\bASAP\\b', 'as soon as possible'],
  ['\\bFAQ\\b', 'F A Q'], ['\\bETA\\b', 'E T A'], ['\\bID\\b', 'I D'], ['\\bOK\\b', 'okay'],
];

const MATH: [string, string][] = [
  ['±', ' plus or minus '], ['≈', ' approximately '], ['≤', ' less than or equal to '],
  ['≥', ' greater than or equal to '], ['≠', ' not equal to '], ['×', ' times '], ['÷', ' divided by '],
  ['√', ' square root of '], ['π', ' pi '], ['∞', ' infinity '], ['½', ' one half '], ['¼', ' one quarter '],
  ['¾', ' three quarters '],
];

function rCode(m: M): string {
  const parts = m.group(0)!.split(/[-_/.]/).filter(Boolean);
  const out: string[] = [];
  for (const p of parts) {
    if (/^[A-Z]{4,}$/.test(p) && /[AEIOU]/.test(p) && p.length <= 8) out.push(p.toLowerCase());
    else if (/^[1-9][0-9]?$/.test(p) && out.length && /^[a-z]+$/.test(out[out.length - 1]!)) out.push(cardinal(p));
    else out.push(findall('[A-Za-z]+|\\d+', p).map((c) => (/^\p{L}+$/u.test(c) ? spell(c) : digits(c))).join(' '));
  }
  return out.join(', ');
}

const FILE_EXTS = new Set([
  'pdf', 'jpg', 'jpeg', 'png', 'gif', 'csv', 'txt', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
  'zip', 'mp3', 'mp4', 'mov', 'wav', 'json', 'html', 'py', 'js', 'exe', 'svg', 'md', 'log', 'tar',
  'gz', 'iso', 'dmg', 'apk', 'heic', 'webp', 'key', 'pages', 'xml', 'yaml', 'yml', 'sh', 'ics',
]);

const MON3 = MONTHS.map((x) => x.slice(0, 3));

/** frontend_en.normalize */
export function normalize(text: string): string {
  let t = ' ' + text.split('\u2212').join('-').split('\u2013').join('-').split('\u2014').join(', ').split('\u2019').join("'") + ' ';
  const last = text.trim().slice(-1);
  const end = '.!?'.includes(last) ? last : '';
  t = sub('\\b(\\d{4})-(\\d{2})-(\\d{2})\\b', (m) => {
    const mo = Number(m.group(2)), d = Number(m.group(3));
    return mo >= 1 && mo <= 12 && d >= 1 && d <= 31 ? dateWords(d, mo, m.group(1)) : m.group(0)!;
  }, t);
  t = sub('\\b(\\d{1,2}:\\d{2})\\s?-\\s?(\\d{1,2}:\\d{2})\\b', '\\1 to \\2', t);
  t = sub('\\b(\\d)/(\\d{1,2})(?=\\s+of\\b)', (m) =>
    FRAC[`${Number(m.group(1))}/${Number(m.group(2))}`] ?? cardinal(m.group(1)!) + ' over ' + cardinal(m.group(2)!), t);
  t = sub('(\\d+(?:\\.\\d+)?)-(inch|foot|meter|metre|mile|liter|litre|hour|minute|day|week|month|year|page|' +
    'person|seat|door|piece|core|bit|megapixel)\\b', (m) => decimal(m.group(1)!) + ' ' + m.group(2), t);
  t = sub('(?<!\\S)(?:[A-Z0-9]{1,6} ){2,}[A-Z0-9]{1,6}(?=[\\s.,;!?]|$)', (m) => {
    const g = m.group(0)!;
    return /[A-Z]/.test(g) && g.replace(/[^0-9]/g, '').length >= 6 ? g.split(/\s+/).filter(Boolean).map(spell).join(', ') : g;
  }, t);
  t = sub('([\\w.+-]+)@([\\w-]+(?:\\.[\\w-]+)+)', rEmail, t);
  t = sub('\\bhttps?://[^\\s,]+|\\bwww\\.[^\\s,]+|\\b(?:[\\w-]+\\.)+(?:com|org|net|io|ai|dev|app|co\\.uk|edu|gov)\\b(?:/[^\\s,]*)?',
    (m) => rUrl(m) + (m.group(0)!.endsWith('.') ? '.' : ''), t);
  t = sub('\\b([\\w\\-]+(?:\\.[\\w\\-]+)*)\\.([A-Za-z]{2,4}\\d?)\\b(?=[\\s,;:!?)]|\\.\\s|\\.$)', (m) => {
    const g1 = m.group(1)!;
    return !fullmatch('\\d+(?:\\.\\d+)*', g1) && (g1.includes('_') || /[0-9]/.test(g1) || FILE_EXTS.has(m.group(2)!.toLowerCase()))
      ? rFile(m) : m.group(0)!;
  }, t);
  t = sub('\\b(v|V|[Vv]ersion |Python |iOS |Android |macOS |Windows |release |build )?(\\d+\\.\\d+(?:\\.\\d+)+)\\b',
    (m) => (m.group(1) || m.group(2)!.split('.').length - 1 >= 2 ? rVersion(m) : m.group(0)!), t);
  t = sub('(?<![\\w-])(v|V)(\\d+(?:\\.\\d+)?)\\b', (m) => 'version ' + m.group(2)!.split('.').map((p) => cardinal(p)).join(' point '), t);
  t = sub('\\b(Python |iOS |Android |macOS |[Vv]ersion )(\\d+)\\.(\\d+)\\b',
    (m) => m.group(1)! + cardinal(m.group(2)!) + ' point ' + cardinal(m.group(3)!), t);
  t = sub('(?<![\\w/\\-])\\+?\\(?\\d{2,5}\\)?(?:[ \\-.]\\d{2,5}){1,4}(?![\\w/:]|-\\w)', (m) => {
    const g = m.group(0)!;
    return g.replace(/[^0-9]/g, '').length >= 7 && !fullmatch('(1[5-9]\\d\\d|20\\d\\d)\\s?-\\s?(1[5-9]\\d\\d|20\\d\\d)', g.trim())
      ? rPhone(m) : g;
  }, t);
  t = sub('(?<![\\d:])(\\d{1,2}):(\\d)(?![\\d:])', (m) => cardinal(m.group(1)!) + ' to ' + cardinal(m.group(2)!), t);
  t = sub('\\b(\\d{1,2}):(\\d{2})\\s*([AaPp]\\.?\\s?[Mm]\\.?)?(?=\\W|$)', rTime, t);
  t = sub('\\b(\\d{1,2})\\s?(?:h|hr|hrs)\\s?(\\d{1,2})?\\s?(min|m)?\\b(?!\\w)', rDuration, t);
  t = sub('\\b(\\d{1,2})/(\\d{1,2})(?:/(\\d{4}|\\d{2}))?\\b', rSlashDate, t);
  t = sub(`\\b(\\d{1,2})\\s?-\\s?(\\d{1,2}) (${MONTHS.join('|')})\\b`,
    (m) => `the ${ordinal(m.group(1)!)} to the ${ordinal(m.group(2)!)} of ${m.group(3)}`, t);
  const mon = MONTHS.join('|') + '|' + MON3.join('|');
  t = sub(`\\b(${mon})\\.? (\\d{1,2})(?:st|nd|rd|th)?(?:,? (\\d{4}))?\\b`,
    (m) => rWrittenMd(MONTHS[MON3.indexOf(m.group(1)!.slice(0, 3))]!, m.group(2)!, m.group(3)), t);
  t = sub(`\\b(\\d{1,2})(?:st|nd|rd|th)? (?:of )?(${MONTHS.join('|')})(?:,? (\\d{4}))?\\b`, rWrittenDm, t);
  t = sub('\\b(?:in|since|from|of|by|until|year|circa|founded in) (1[1-9]\\d\\d|20\\d\\d)\\b',
    (m) => m.group(0)!.slice(0, -4) + year(m.group(1)!), t);
  t = sub('\\b(US|A|C|NZ|HK)?\\$\\s?(\\d[\\d,]*(?:\\.\\d+)?) (thousand|million|billion|trillion)\\b',
    (m) => decimal(m.group(2)!) + ' ' + m.group(3) + (m.group(1) ? ' ' + spell(m.group(1)!) : '') + ' dollars', t);
  t = sub('\\b(US|A|C|NZ|HK)\\$(?=\\s?\\d)', (m) => spell(m.group(1)!) + ' $', t);
  t = sub('(-?)([$€£¥])\\s?(\\d[\\d,]*(?:\\.\\d+)?)(?:\\s?(bn|mn|[kKmMbB])\\b)?',
    (m) => (m.group(1) ? 'minus ' : '') + rCurrency(m.group(2)!, m.group(3)!, m.group(4)), t);
  t = sub('(-?\\d+(?:\\.\\d+)?)\\s?%', rPercent, t);
  t = sub('(-?\\d+(?:\\.\\d+)?)\\s?°\\s?([CF])?\\b', rTemp, t);
  t = sub('#\\s?(\\d+(?:-\\d+)+)\\b', (m) => 'number ' + m.group(1)!.split('-').map(digits).join(', '), t);
  t = sub(`(?<![\\w.\\-])(\\d[\\d,]*(?:\\.\\d+)?)\\s?-\\s?(\\d[\\d,]*(?:\\.\\d+)?)(?:\\s?(${UNIT_RE})(?![\\w/]))?(?![\\w.]*\\d|-\\w)`, rRange, t);
  t = sub('#\\s?(\\d[\\d,]*)', (m) => 'number ' + number(m.group(1)!), t);
  t = sub('#(?=[A-Za-z])', 'number ', t);
  t = sub('\\b(\\d[\\d,]*)(?:st|nd|rd|th)\\b', (m) => ordinal(m.group(1)!.replace(/,/g, '')), t);
  t = sub(`(?<![\\w.])(\\d[\\d,]*(?:\\.\\d+)?)\\s?(${UNIT_RE})(?![\\w/])`, rUnit, t);
  for (const [pat, rep] of ABBR) t = sub(pat, rep, t);
  for (const [a, b] of MATH) t = t.split(a).join(b);
  t = sub('(?<=\\d)\\s?[x×]\\s?(?=\\d)', ' by ', t);
  t = sub('(?<=\\s)\\+(?=\\s|\\d)', 'plus ', t);
  t = sub('(?<=\\s)=(?=\\s)', 'equals', t);
  t = sub('(?<=\\d)\\s?/\\s?(?=\\d)', ' over ', t);
  t = sub('\\s<\\s', ' less than ', t);
  t = sub('\\s>\\s', ' greater than ', t);
  t = sub('(?<![\\w-])(\\d{1,3})([A-Z])(?![\\w-])', (m) => cardinal(m.group(1)!) + ' ' + m.group(2), t);
  t = sub('\\b(?=[A-Za-z0-9\\-]*\\d)(?=[A-Za-z0-9\\-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\\b', rCode, t);
  t = sub('\\b[A-Z]{2,}(?:-[A-Z0-9]+)+\\b', rCode, t);
  t = sub('\\d[\\d,]*(?:\\.\\d+)?', (m) => number(m.group(0)!), t);
  t = t.split('&').join(' and ').split('@').join(' at ').split('_').join(' underscore ');
  t = sub('(?<=\\w)/(?=\\w)', ' slash ', t);
  t = sub('\\s+', ' ', t).trim();
  t = sub('\\s+([,.;:!?])', '\\1', t);
  t = sub('\\b([Tt]he) the\\b', '\\1', t);
  if (end && !t.endsWith(end) && !'.!?'.includes(t.slice(-1))) t += end;
  return t;
}
