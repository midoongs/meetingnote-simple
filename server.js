import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
try { process.loadEnvFile(path.join(ROOT, '.env')); } catch {}

const API_KEY = process.env.GEMINI_API_KEY;
const MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite';
const PORT = process.env.PORT || 3000;
const MAX_BYTES = 200 * 1024 * 1024;
const INLINE_LIMIT = 15 * 1024 * 1024; // 이보다 크면 Files API 사용
const BASE = 'https://generativelanguage.googleapis.com';
const MAX_ATTEMPTS = 5;
const DATA_DIR = path.join(ROOT, 'data');
const AUDIO_DIR = path.join(DATA_DIR, 'audio');

if (!API_KEY) { console.error('GEMINI_API_KEY 가 .env 에 없습니다.'); process.exit(1); }
await fs.mkdir(AUDIO_DIR, { recursive: true });

const sleep = ms => new Promise(r => setTimeout(r, ms));
const metaPath = id => path.join(DATA_DIR, `${id}.json`);
const safeId = id => /^[a-f0-9-]{36}$/.test(id);

async function save(m) { await fs.writeFile(metaPath(m.id), JSON.stringify(m, null, 2)); }
async function load(id) {
  try { return JSON.parse(await fs.readFile(metaPath(id), 'utf8')); } catch { return null; }
}

const SCHEMA = {
  type: 'OBJECT',
  properties: {
    title: { type: 'STRING' },
    transcript: { type: 'STRING' },
    summary: { type: 'STRING' },
    decisions: { type: 'ARRAY', items: { type: 'STRING' } },
    todos: {
      type: 'ARRAY',
      items: {
        type: 'OBJECT',
        properties: { task: { type: 'STRING' }, owner: { type: 'STRING' }, due: { type: 'STRING' } },
        required: ['task'],
      },
    },
  },
  required: ['title', 'transcript', 'summary', 'decisions', 'todos'],
};

const PROMPT = `첨부된 회의 녹취를 한국어로 정리하세요.
- title: 회의를 한 줄로 나타내는 제목
- transcript: 전체 받아쓰기 (화자가 구분되면 "화자1:" 형식으로 줄바꿈)
- summary: 핵심 내용 요약 (3~6문장)
- decisions: 회의에서 확정된 결정사항 목록 (없으면 빈 배열)
- todos: 할 일 목록. task=할 일, owner=담당자(모르면 빈 문자열), due=기한(모르면 빈 문자열)
owner 에는 사람 이름(또는 직함)만, due 에는 녹취에 기한이 언급되면 반드시 그 표현(예: 다음 주 금요일)만 짧게 적고 설명·주석·괄호를 붙이지 마세요. 필드 안에 다른 필드 이름을 쓰지 마세요.
녹취에 없는 내용은 지어내지 마세요.`;

class RetryableError extends Error {}

async function uploadFile(buf, mime) {
  const start = await fetch(`${BASE}/upload/v1beta/files`, {
    method: 'POST',
    headers: {
      'x-goog-api-key': API_KEY, 'X-Goog-Upload-Protocol': 'resumable', 'X-Goog-Upload-Command': 'start',
      'X-Goog-Upload-Header-Content-Length': String(buf.length), 'X-Goog-Upload-Header-Content-Type': mime,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ file: { display_name: 'meeting' } }),
  });
  if (!start.ok) throw new RetryableError(`파일 업로드 시작 실패 ${start.status}`);
  const up = await fetch(start.headers.get('x-goog-upload-url'), {
    method: 'POST',
    headers: { 'X-Goog-Upload-Offset': '0', 'X-Goog-Upload-Command': 'upload, finalize' },
    body: buf,
  });
  if (!up.ok) throw new RetryableError(`파일 업로드 실패 ${up.status}`);
  let file = (await up.json()).file;
  for (let i = 0; i < 60 && file.state === 'PROCESSING'; i++) {
    await sleep(2000);
    file = await (await fetch(`${BASE}/v1beta/${file.name}`, { headers: { 'x-goog-api-key': API_KEY } })).json();
  }
  if (file.state !== 'ACTIVE') throw new RetryableError(`파일 처리 실패: ${file.state}`);
  return file;
}

async function callGemini(buf, mime) {
  let res;
  try {
    const media = buf.length > INLINE_LIMIT
      ? (f => ({ file_data: { mime_type: mime, file_uri: f.uri } }))(await uploadFile(buf, mime))
      : { inline_data: { mime_type: mime, data: buf.toString('base64') } };
    res = await fetch(`${BASE}/v1beta/models/${MODEL}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': API_KEY },
      body: JSON.stringify({
        contents: [{ parts: [media, { text: PROMPT }] }],
        generationConfig: { responseMimeType: 'application/json', responseSchema: SCHEMA },
      }),
      signal: AbortSignal.timeout(10 * 60 * 1000),
    });
  } catch (e) {
    if (e instanceof RetryableError) throw e;
    throw new RetryableError(`네트워크 오류: ${e.message}`);
  }
  if (!res.ok) {
    const text = await res.text();
    const msg = `Gemini ${res.status}: ${text.slice(0, 300)}`;
    if (res.status === 429 || res.status >= 500) throw new RetryableError(msg);
    throw new Error(msg);
  }
  const json = await res.json();
  const out = json.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('');
  if (!out) throw new RetryableError('Gemini 응답이 비어 있습니다.');
  try { return JSON.parse(out); } catch { throw new RetryableError('응답 JSON 파싱 실패'); }
}

const running = new Set();

async function transcribe(id) {
  if (running.has(id)) return;
  running.add(id);
  try {
    const m = await load(id);
    const audioFile = path.join(AUDIO_DIR, id);
    const buf = await fs.readFile(audioFile);
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      m.status = 'processing'; m.attempt = attempt; await save(m);
      try {
        const r = await callGemini(buf, m.mime);
        Object.assign(m, r, { status: 'done', error: null, waitSec: 0 });
        await save(m);
        await fs.rm(audioFile, { force: true });
        return;
      } catch (e) {
        m.error = e.message;
        if (!(e instanceof RetryableError) || attempt === MAX_ATTEMPTS) {
          m.status = 'failed'; await save(m); return;
        }
        const wait = Math.min(60, 5 * 2 ** (attempt - 1)); // 5,10,20,40초
        m.status = 'waiting'; m.waitSec = wait; await save(m);
        await sleep(wait * 1000);
      }
    }
  } catch (e) {
    const m = await load(id);
    if (m) { m.status = 'failed'; m.error = e.message; await save(m); }
  } finally { running.delete(id); }
}

// 서버 재시작 시 중단된 작업 이어서 처리
for (const f of await fs.readdir(DATA_DIR)) {
  if (!f.endsWith('.json')) continue;
  const m = await load(f.slice(0, -5));
  if (m && ['processing', 'waiting'].includes(m.status)) transcribe(m.id);
}

const send = (res, code, body) => {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
};

async function readBody(req) {
  const chunks = []; let size = 0;
  for await (const c of req) {
    size += c.length;
    if (size > MAX_BYTES) throw Object.assign(new Error('파일이 너무 큽니다 (최대 200MB).'), { code: 413 });
    chunks.push(c);
  }
  return Buffer.concat(chunks);
}

http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://x');
    const p = url.pathname;

    if (req.method === 'GET' && (p === '/' || p === '/index.html')) {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      return res.end(await fs.readFile(path.join(ROOT, 'public', 'index.html')));
    }
    if (req.method === 'GET' && p === '/api/meetings') {
      const list = [];
      for (const f of await fs.readdir(DATA_DIR)) {
        if (!f.endsWith('.json')) continue;
        const { transcript, ...rest } = (await load(f.slice(0, -5))) || {};
        if (rest.id) list.push(rest);
      }
      list.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
      return send(res, 200, list);
    }
    if (req.method === 'POST' && p === '/api/meetings') {
      const buf = await readBody(req);
      if (!buf.length) return send(res, 400, { error: '빈 파일입니다.' });
      const id = randomUUID();
      const name = decodeURIComponent(req.headers['x-filename'] || '녹취');
      const mime = 'audio/wav';
      await fs.writeFile(path.join(AUDIO_DIR, id), buf);
      const m = { id, title: name, fileName: name, mime, status: 'processing', attempt: 0, createdAt: new Date().toISOString() };
      await save(m);
      transcribe(id);
      return send(res, 202, { id });
    }

    const mm = p.match(/^\/api\/meetings\/([^/]+)(\/retry)?$/);
    if (mm && safeId(mm[1])) {
      const id = mm[1];
      const m = await load(id);
      if (!m) return send(res, 404, { error: '없는 회의록입니다.' });
      if (req.method === 'GET' && !mm[2]) return send(res, 200, m);
      if (req.method === 'DELETE' && !mm[2]) {
        await fs.rm(metaPath(id), { force: true });
        await fs.rm(path.join(AUDIO_DIR, id), { force: true });
        return send(res, 200, { ok: true });
      }
      if (req.method === 'POST' && mm[2]) {
        if (m.status !== 'failed') return send(res, 400, { error: '실패한 건만 재시도할 수 있습니다.' });
        transcribe(id);
        return send(res, 202, { ok: true });
      }
    }
    send(res, 404, { error: 'Not found' });
  } catch (e) {
    send(res, e.code === 413 ? 413 : 500, { error: e.message });
  }
}).listen(PORT, () => console.log(`회의록 정리기: http://localhost:${PORT}  (model: ${MODEL})`));
