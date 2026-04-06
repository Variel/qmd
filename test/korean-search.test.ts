import { afterEach, describe, expect, test } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { rebuildKoreanSearchShadowIndex } from "../src/korean-search.js";
import { createStore } from "../src/store.js";

const cleanupPaths: string[] = [];

afterEach(() => {
  for (const path of cleanupPaths.splice(0)) {
    rmSync(path, { recursive: true, force: true });
  }
});

function createTestStore() {
  const dir = mkdtempSync(join(tmpdir(), "qmd-korean-search-"));
  cleanupPaths.push(dir);
  return createStore(join(dir, "index.sqlite"));
}

describe("Korean-aware shadow FTS", () => {
  test("searchFTS can match Korean compound terms through the shadow index", async () => {
    const store = createTestStore();
    const now = new Date().toISOString();
    const hash = "hash-korean-1";
    const body = "# 감사 로그\n\n보안취약점 스캔 결과를 감사 로그에 저장합니다.\n";

    try {
      store.insertContent(hash, body, now);
      store.insertDocument("docs", "audit.md", "감사 로그", hash, now, now);

      expect(store.searchFTS("취약점", 5)).toEqual([]);

      await rebuildKoreanSearchShadowIndex(store.db, {
        tokenize: async (text) =>
          text.replace("보안취약점", "보안취약점 보안 취약점"),
      });

      const results = store.searchFTS("취약점", 5);
      expect(results).toHaveLength(1);
      expect(results[0]?.displayPath).toBe("docs/audit.md");
      expect(results[0]?.title).toBe("감사 로그");
    } finally {
      store.close();
    }
  });
});
