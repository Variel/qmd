import type { Database } from "./db.js";
import type { SearchResult } from "./store.js";
import { buildShadowProjectionText, containsHangul, type KiwiTokenizerDependencies } from "./korean.js";

const QMD_KOREAN_SEARCH_POLICY_ID = "kiwi-cong-shadow-v1";
const QMD_KOREAN_SEARCH_POLICY_METADATA_KEY = "qmd_korean_search_policy_id";
const QMD_KOREAN_SEARCH_SOURCE_SNAPSHOT_METADATA_KEY = "qmd_korean_search_source_snapshot";
const QMD_KOREAN_SEARCH_SHADOW_TABLE = "qmd_documents_fts_ko";

type RebuildRow = {
  readonly id: number;
  readonly collection: string;
  readonly path: string;
  readonly title: string;
  readonly body: string;
  readonly modified_at?: string;
};

type ShadowSearchRow = {
  readonly filepath: string;
  readonly display_path: string;
  readonly title: string;
  readonly body: string;
  readonly hash: string;
  readonly modified_at: string;
  readonly collection: string;
  readonly bm25_score: number;
};

export interface SearchSourceSnapshot {
  readonly totalDocuments: number;
  readonly latestModifiedAt?: string;
  readonly maxDocumentId?: number;
}

export interface KoreanSearchShadowIndexDependencies {
  readonly tokenize?: (text: string, dependencies?: KiwiTokenizerDependencies) => Promise<string>;
  readonly kiwiDependencies?: KiwiTokenizerDependencies;
}

function sanitizeFTS5Term(term: string): string {
  return term.replace(/[^\p{L}\p{N}']/gu, "").toLowerCase();
}

function buildFTS5Query(query: string): string | null {
  const positive: string[] = [];
  const negative: string[] = [];
  let index = 0;
  const source = query.trim();

  while (index < source.length) {
    while (index < source.length && /\s/.test(source[index] ?? "")) {
      index += 1;
    }
    if (index >= source.length) break;

    const negated = source[index] === "-";
    if (negated) index += 1;

    if (source[index] === '"') {
      const start = index + 1;
      index += 1;
      while (index < source.length && source[index] !== '"') {
        index += 1;
      }

      const phrase = source.slice(start, index).trim();
      index += 1;
      if (!phrase) continue;

      const sanitized = phrase
        .split(/\s+/)
        .map((term) => sanitizeFTS5Term(term))
        .filter(Boolean)
        .join(" ");
      if (!sanitized) continue;

      const ftsPhrase = `"${sanitized}"`;
      if (negated) negative.push(ftsPhrase);
      else positive.push(ftsPhrase);
      continue;
    }

    const start = index;
    while (index < source.length && !/[\s"]/.test(source[index] ?? "")) {
      index += 1;
    }

    const sanitized = sanitizeFTS5Term(source.slice(start, index));
    if (!sanitized) continue;

    const ftsTerm = `"${sanitized}"*`;
    if (negated) negative.push(ftsTerm);
    else positive.push(ftsTerm);
  }

  if (positive.length === 0) return null;

  let result = positive.join(" AND ");
  for (const term of negative) {
    result = `${result} NOT ${term}`;
  }
  return result;
}

function toDocId(hash: string): string {
  return hash.slice(0, 6);
}

export function ensureKoreanSearchShadowTable(db: Database): void {
  db.exec(
    [
      `CREATE VIRTUAL TABLE IF NOT EXISTS ${QMD_KOREAN_SEARCH_SHADOW_TABLE} USING fts5(`,
      "  filepath,",
      "  title,",
      "  body,",
      "  tokenize='porter unicode61'",
      ")",
    ].join("\n"),
  );
}

function listActiveDocuments(db: Database): RebuildRow[] {
  return db.prepare(
    [
      "SELECT d.id, d.collection, d.path, d.title, c.doc AS body, d.modified_at",
      "FROM documents d",
      "JOIN content c ON c.hash = d.hash",
      "WHERE d.active = 1",
      "ORDER BY d.id ASC",
    ].join("\n"),
  ).all() as RebuildRow[];
}

function buildSearchSourceSnapshot(rows: readonly RebuildRow[]): SearchSourceSnapshot {
  let latestModifiedAt: string | undefined;
  let maxDocumentId: number | undefined;

  for (const row of rows) {
    if (latestModifiedAt === undefined || (row.modified_at && row.modified_at > latestModifiedAt)) {
      latestModifiedAt = row.modified_at;
    }
    if (maxDocumentId === undefined || row.id > maxDocumentId) {
      maxDocumentId = row.id;
    }
  }

  return {
    totalDocuments: rows.length,
    latestModifiedAt,
    maxDocumentId,
  };
}

function readCurrentSearchSourceSnapshot(db: Database): SearchSourceSnapshot {
  const row = db.prepare(
    [
      "SELECT",
      "  COUNT(*) AS count,",
      "  MAX(d.modified_at) AS latest_modified_at,",
      "  MAX(d.id) AS max_document_id",
      "FROM documents d",
      "WHERE d.active = 1",
    ].join("\n"),
  ).get() as
    | { count?: number; latest_modified_at?: string; max_document_id?: number }
    | undefined;

  return {
    totalDocuments: row?.count ?? 0,
    latestModifiedAt: row?.latest_modified_at,
    maxDocumentId: row?.max_document_id,
  };
}

function readStoredSearchSourceSnapshot(db: Database): SearchSourceSnapshot | undefined {
  const row = db.prepare(
    "SELECT value FROM store_config WHERE key = ?",
  ).get(QMD_KOREAN_SEARCH_SOURCE_SNAPSHOT_METADATA_KEY) as { value?: string } | undefined;

  if (typeof row?.value !== "string") return undefined;

  try {
    const parsed = JSON.parse(row.value) as Partial<SearchSourceSnapshot>;
    if (typeof parsed.totalDocuments !== "number") return undefined;
    return {
      totalDocuments: parsed.totalDocuments,
      latestModifiedAt: typeof parsed.latestModifiedAt === "string" ? parsed.latestModifiedAt : undefined,
      maxDocumentId: typeof parsed.maxDocumentId === "number" ? parsed.maxDocumentId : undefined,
    };
  } catch {
    return undefined;
  }
}

function readStoredPolicyId(db: Database): string | undefined {
  const row = db.prepare(
    "SELECT value FROM store_config WHERE key = ?",
  ).get(QMD_KOREAN_SEARCH_POLICY_METADATA_KEY) as { value?: string } | undefined;
  return typeof row?.value === "string" ? row.value : undefined;
}

function shadowTableExists(db: Database): boolean {
  const row = db.prepare(
    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
  ).get(QMD_KOREAN_SEARCH_SHADOW_TABLE) as { name?: string } | undefined;
  return typeof row?.name === "string";
}

function countIndexedDocuments(db: Database): number {
  if (!shadowTableExists(db)) return 0;
  const row = db.prepare(
    [
      "SELECT COUNT(*) AS count",
      `FROM ${QMD_KOREAN_SEARCH_SHADOW_TABLE} f`,
      "JOIN documents d ON d.id = f.rowid",
      "WHERE d.active = 1",
    ].join("\n"),
  ).get() as { count?: number } | undefined;

  return row?.count ?? 0;
}

function snapshotsMatch(a?: SearchSourceSnapshot, b?: SearchSourceSnapshot): boolean {
  return !!a && !!b
    && a.totalDocuments === b.totalDocuments
    && a.latestModifiedAt === b.latestModifiedAt
    && a.maxDocumentId === b.maxDocumentId;
}

export function isKoreanSearchShadowIndexFresh(db: Database): boolean {
  const current = readCurrentSearchSourceSnapshot(db);
  const stored = readStoredSearchSourceSnapshot(db);
  const indexedDocuments = countIndexedDocuments(db);
  const storedPolicy = readStoredPolicyId(db);

  if (!shadowTableExists(db)) return current.totalDocuments === 0;
  if (storedPolicy !== QMD_KOREAN_SEARCH_POLICY_ID) return false;
  if (!snapshotsMatch(current, stored)) return false;
  if (indexedDocuments < current.totalDocuments) return false;
  return true;
}

export function shouldUseKoreanSearchShadowIndex(db: Database, query: string): boolean {
  if (!containsHangul(query)) return false;
  return isKoreanSearchShadowIndexFresh(db);
}

export async function rebuildKoreanSearchShadowIndex(
  db: Database,
  dependencies: KoreanSearchShadowIndexDependencies = {},
): Promise<{
  indexedDocuments: number;
  totalDurationMs: number;
  sourceSnapshot: SearchSourceSnapshot;
}> {
  const totalStart = Date.now();
  const tokenize =
    dependencies.tokenize ??
    ((text: string, kiwiDependencies?: KiwiTokenizerDependencies) =>
      buildShadowProjectionText(text, kiwiDependencies));
  const rows = listActiveDocuments(db);
  const sourceSnapshot = buildSearchSourceSnapshot(rows);

  const projections = await Promise.all(
    rows.map(async (row) => ({
      rowId: row.id,
      filepath: await tokenize(`${row.collection}/${row.path}`, dependencies.kiwiDependencies),
      title: await tokenize(row.title, dependencies.kiwiDependencies),
      body: await tokenize(row.body, dependencies.kiwiDependencies),
    })),
  );

  db.exec("BEGIN IMMEDIATE");
  try {
    ensureKoreanSearchShadowTable(db);
    db.prepare(`DELETE FROM ${QMD_KOREAN_SEARCH_SHADOW_TABLE}`).run();

    const insert = db.prepare(
      `INSERT INTO ${QMD_KOREAN_SEARCH_SHADOW_TABLE}(rowid, filepath, title, body) VALUES (?, ?, ?, ?)`,
    );
    for (const projection of projections) {
      insert.run(projection.rowId, projection.filepath, projection.title, projection.body);
    }

    db.prepare(
      [
        "INSERT INTO store_config (key, value) VALUES (?, ?)",
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      ].join("\n"),
    ).run(QMD_KOREAN_SEARCH_POLICY_METADATA_KEY, QMD_KOREAN_SEARCH_POLICY_ID);
    db.prepare(
      [
        "INSERT INTO store_config (key, value) VALUES (?, ?)",
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      ].join("\n"),
    ).run(QMD_KOREAN_SEARCH_SOURCE_SNAPSHOT_METADATA_KEY, JSON.stringify(sourceSnapshot));

    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }

  return {
    indexedDocuments: rows.length,
    totalDurationMs: Date.now() - totalStart,
    sourceSnapshot,
  };
}

export function searchKoreanShadowIndex(
  db: Database,
  query: string,
  limit: number = 20,
  collectionName?: string,
): SearchResult[] {
  const ftsQuery = buildFTS5Query(query);
  if (!ftsQuery) return [];

  const params: (string | number)[] = [ftsQuery];
  const ftsLimit = collectionName ? limit * 10 : limit;

  let sql = [
    "WITH fts_matches AS (",
    `  SELECT rowid, bm25(${QMD_KOREAN_SEARCH_SHADOW_TABLE}, 10.0, 1.0) AS bm25_score`,
    `  FROM ${QMD_KOREAN_SEARCH_SHADOW_TABLE}`,
    `  WHERE ${QMD_KOREAN_SEARCH_SHADOW_TABLE} MATCH ?`,
    "  ORDER BY bm25_score ASC",
    `  LIMIT ${ftsLimit}`,
    ")",
    "SELECT",
    "  'qmd://' || d.collection || '/' || d.path AS filepath,",
    "  d.collection || '/' || d.path AS display_path,",
    "  d.title,",
    "  content.doc AS body,",
    "  d.hash,",
    "  d.modified_at,",
    "  d.collection,",
    "  fm.bm25_score",
    "FROM fts_matches fm",
    "JOIN documents d ON d.id = fm.rowid",
    "JOIN content ON content.hash = d.hash",
    "WHERE d.active = 1",
  ].join("\n");

  if (collectionName) {
    sql += "\nAND d.collection = ?";
    params.push(collectionName);
  }

  sql += "\nORDER BY fm.bm25_score ASC LIMIT ?";
  params.push(limit);

  const rows = db.prepare(sql).all(...params) as ShadowSearchRow[];
  return rows.map((row) => {
    const score = Math.abs(row.bm25_score) / (1 + Math.abs(row.bm25_score));
    return {
      filepath: row.filepath,
      displayPath: row.display_path,
      title: row.title,
      context: null,
      hash: row.hash,
      docid: toDocId(row.hash),
      collectionName: row.collection,
      modifiedAt: row.modified_at,
      bodyLength: row.body.length,
      body: row.body,
      score,
      source: "fts" as const,
    };
  });
}
