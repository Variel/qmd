import {
  isOpenAIEmbeddingModel,
  isVoyageRerankModel,
  normalizeEmbeddingModelUri,
  normalizeRerankModelUri,
} from "../llm.js";
import type { ApiUsageSummary } from "./types.js";

export type ApiUsageRecord = {
  provider: "openai" | "voyage";
  endpoint: "embeddings" | "rerank";
  model: string;
  input_tokens: number;
  estimated_cost_usd: number;
  url: string;
};

const DEFAULT_PRICE_PER_1M_TOKENS: Record<string, number> = {
  "text-embedding-3-small": 0.02,
  "text-embedding-3-large": 0.13,
  "text-embedding-ada-002": 0.10,
  "rerank-2.5": 0.05,
  "rerank-2.5-lite": 0.02,
};

function priceOverrideEnvKey(model: string): string {
  const normalized = model
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return `QMD_BENCH_PRICE_${normalized}_PER_1M_TOKENS`;
}

export function getPricePer1MTokens(model: string, env: NodeJS.ProcessEnv = process.env): number | null {
  const envKey = priceOverrideEnvKey(model);
  const override = env[envKey];
  if (override !== undefined) {
    const parsed = Number.parseFloat(override);
    if (Number.isFinite(parsed) && parsed >= 0) {
      return parsed;
    }
  }

  return DEFAULT_PRICE_PER_1M_TOKENS[model] ?? null;
}

export function estimateCostUsd(model: string, inputTokens: number, env: NodeJS.ProcessEnv = process.env): number {
  const pricePerMillion = getPricePer1MTokens(model, env);
  if (pricePerMillion === null) return 0;
  return (inputTokens / 1_000_000) * pricePerMillion;
}

export function estimateTextTokens(text: string): number {
  if (!text) return 0;

  const hangulCount = (text.match(/[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]/g) ?? []).length;
  const asciiWordCount = (text.match(/[A-Za-z0-9]+(?:[-_./:][A-Za-z0-9]+)*/g) ?? []).length;
  const punctuationCount = (text.match(/[^\sA-Za-z0-9\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]/g) ?? []).length;

  return hangulCount + asciiWordCount + Math.ceil(punctuationCount * 0.5);
}

function parseJsonBody(body: unknown): any | null {
  if (typeof body !== "string") return null;
  try {
    return JSON.parse(body);
  } catch {
    return null;
  }
}

function extractOpenAIEmbeddingTokens(payload: any, responseJson: any): number {
  const fromUsage = responseJson?.usage?.total_tokens;
  if (typeof fromUsage === "number") return fromUsage;

  const inputs = Array.isArray(payload?.input) ? payload.input : [payload?.input];
  return inputs
    .filter((item) => typeof item === "string")
    .reduce((total, item) => total + estimateTextTokens(item), 0);
}

function extractVoyageRerankTokens(payload: any, responseJson: any): number {
  const directUsage = responseJson?.total_tokens ?? responseJson?.usage?.total_tokens;
  if (typeof directUsage === "number") return directUsage;

  const query = typeof payload?.query === "string" ? payload.query : "";
  const queryTokens = estimateTextTokens(query);
  const docs = Array.isArray(payload?.documents)
    ? payload.documents.filter((item: unknown): item is string => typeof item === "string")
    : [];
  const docTokens = docs.reduce((total: number, doc: string) => total + estimateTextTokens(doc), 0);

  return queryTokens * Math.max(docs.length, 1) + docTokens;
}

function resolveUrl(input: Request | URL | string): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return input.url;
}

function emptyUsageSummary(): ApiUsageSummary {
  return {
    request_count: 0,
    input_tokens: 0,
    estimated_cost_usd: 0,
    by_model: {},
  };
}

export class ApiUsageTracker {
  readonly records: ApiUsageRecord[] = [];

  record(record: Omit<ApiUsageRecord, "estimated_cost_usd">): void {
    this.records.push({
      ...record,
      estimated_cost_usd: estimateCostUsd(record.model, record.input_tokens),
    });
  }

  summarize(): ApiUsageSummary {
    const summary = emptyUsageSummary();

    for (const record of this.records) {
      summary.request_count += 1;
      summary.input_tokens += record.input_tokens;
      summary.estimated_cost_usd += record.estimated_cost_usd;

      const byModel = summary.by_model[record.model] ?? {
        request_count: 0,
        input_tokens: 0,
        estimated_cost_usd: 0,
      };
      byModel.request_count += 1;
      byModel.input_tokens += record.input_tokens;
      byModel.estimated_cost_usd += record.estimated_cost_usd;
      summary.by_model[record.model] = byModel;
    }

    return summary;
  }
}

export async function withApiUsageTracking<T>(
  tracker: ApiUsageTracker,
  fn: () => Promise<T>,
): Promise<T> {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (input: Request | URL | string, init?: RequestInit): Promise<Response> => {
    const response = await originalFetch(input, init);
    const url = resolveUrl(input);
    const requestBody = parseJsonBody(init?.body);

    try {
      const responseJson = await response.clone().json();

      if (url.endsWith("/embeddings")) {
        const model = normalizeEmbeddingModelUri(String(requestBody?.model ?? ""));
        if (isOpenAIEmbeddingModel(model)) {
          tracker.record({
            provider: "openai",
            endpoint: "embeddings",
            model,
            input_tokens: extractOpenAIEmbeddingTokens(requestBody, responseJson),
            url,
          });
        }
      } else if (url.endsWith("/rerank")) {
        const model = normalizeRerankModelUri(String(requestBody?.model ?? ""));
        if (isVoyageRerankModel(model)) {
          tracker.record({
            provider: "voyage",
            endpoint: "rerank",
            model,
            input_tokens: extractVoyageRerankTokens(requestBody, responseJson),
            url,
          });
        }
      }
    } catch {
      // Ignore non-JSON or unparseable responses — benchmark still proceeds.
    }

    return response;
  };

  try {
    return await fn();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

export function mergeUsageSummaries(...summaries: ApiUsageSummary[]): ApiUsageSummary {
  const merged = emptyUsageSummary();

  for (const summary of summaries) {
    merged.request_count += summary.request_count;
    merged.input_tokens += summary.input_tokens;
    merged.estimated_cost_usd += summary.estimated_cost_usd;

    for (const [model, byModel] of Object.entries(summary.by_model)) {
      const existing = merged.by_model[model] ?? {
        request_count: 0,
        input_tokens: 0,
        estimated_cost_usd: 0,
      };
      existing.request_count += byModel.request_count;
      existing.input_tokens += byModel.input_tokens;
      existing.estimated_cost_usd += byModel.estimated_cost_usd;
      merged.by_model[model] = existing;
    }
  }

  return merged;
}
