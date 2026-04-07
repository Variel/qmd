import { describe, test, expect } from "vitest";
import {
  ApiUsageTracker,
  estimateCostUsd,
  estimateTextTokens,
  getPricePer1MTokens,
  mergeUsageSummaries,
} from "../src/bench/cost.js";

describe("bench cost helpers", () => {
  test("uses default public pricing when no override is present", () => {
    expect(getPricePer1MTokens("text-embedding-3-small")).toBe(0.02);
    expect(getPricePer1MTokens("rerank-2.5-lite")).toBe(0.02);
  });

  test("respects environment pricing overrides", () => {
    const prev = process.env.QMD_BENCH_PRICE_RERANK_2_5_LITE_PER_1M_TOKENS;
    process.env.QMD_BENCH_PRICE_RERANK_2_5_LITE_PER_1M_TOKENS = "0.123";
    try {
      expect(getPricePer1MTokens("rerank-2.5-lite")).toBe(0.123);
    } finally {
      if (prev === undefined) delete process.env.QMD_BENCH_PRICE_RERANK_2_5_LITE_PER_1M_TOKENS;
      else process.env.QMD_BENCH_PRICE_RERANK_2_5_LITE_PER_1M_TOKENS = prev;
    }
  });

  test("estimates token counts for mixed Korean and API-heavy text", () => {
    expect(estimateTextTokens("보안취약점 JWT /login")).toBeGreaterThan(3);
  });

  test("computes cost from token counts", () => {
    expect(estimateCostUsd("text-embedding-3-small", 50_000)).toBeCloseTo(0.001, 6);
    expect(estimateCostUsd("rerank-2.5", 50_000)).toBeCloseTo(0.0025, 6);
  });

  test("aggregates and merges usage summaries", () => {
    const trackerA = new ApiUsageTracker();
    trackerA.record({
      provider: "openai",
      endpoint: "embeddings",
      model: "text-embedding-3-small",
      input_tokens: 10_000,
      url: "https://api.openai.com/v1/embeddings",
    });

    const trackerB = new ApiUsageTracker();
    trackerB.record({
      provider: "voyage",
      endpoint: "rerank",
      model: "rerank-2.5-lite",
      input_tokens: 20_000,
      url: "https://api.voyageai.com/v1/rerank",
    });

    const merged = mergeUsageSummaries(trackerA.summarize(), trackerB.summarize());
    expect(merged.request_count).toBe(2);
    expect(merged.input_tokens).toBe(30_000);
    expect(merged.by_model["text-embedding-3-small"]?.request_count).toBe(1);
    expect(merged.by_model["rerank-2.5-lite"]?.request_count).toBe(1);
    expect(merged.estimated_cost_usd).toBeCloseTo(0.0006, 6);
  });
});
