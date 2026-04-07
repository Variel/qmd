import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import {
  createStore,
  type CollectionConfig,
  type QMDStore,
  type HybridQueryResult,
  type SearchResult,
} from "../index.js";
import { loadConfig } from "../collections.js";
import { withApiUsageTracking, ApiUsageTracker, mergeUsageSummaries } from "./cost.js";
import { scoreResults } from "./score.js";
import type {
  BenchmarkProfile,
  BenchmarkQuery,
  ComparisonBenchmarkPlan,
  ComparisonBenchmarkResult,
  ComparisonQueryResult,
  ProfileComparisonResult,
} from "./types.js";

function deepClone<T>(value: T): T {
  return structuredClone(value);
}

function resolveMaybeRelative(baseDir: string, value: string): string {
  return isAbsolute(value) ? value : resolve(baseDir, value);
}

function resolvePlanConfig(plan: ComparisonBenchmarkPlan, planPath: string): CollectionConfig {
  const planDir = dirname(resolve(planPath));
  const baseConfig = plan.config ? deepClone(plan.config) : deepClone(loadConfig());

  if (!baseConfig.collections || Object.keys(baseConfig.collections).length === 0) {
    throw new Error(
      "Comparison benchmark needs collections. Add them to the plan's config block or configure qmd collections first.",
    );
  }

  for (const collection of Object.values(baseConfig.collections)) {
    if (collection.path) {
      collection.path = resolveMaybeRelative(planDir, collection.path);
    }
  }

  return baseConfig;
}

function sanitizeProfileName(name: string): string {
  return name.replace(/[^A-Za-z0-9._-]+/g, "-");
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * p) - 1));
  return sorted[index] ?? 0;
}

async function runMode(
  store: QMDStore,
  profile: BenchmarkProfile,
  query: string,
  limit: number,
  collection?: string,
): Promise<string[]> {
  const minScore = profile.min_score;

  switch (profile.mode) {
    case "bm25": {
      const results = await store.searchLex(query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    }
    case "vector": {
      const results = await store.searchVector(query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    }
    case "hybrid": {
      const results = await store.search({
        query,
        limit,
        collection,
        minScore,
        rerank: profile.rerank ?? false,
      });
      return results.map((r: HybridQueryResult) => r.file);
    }
    case "full":
    default: {
      const results = await store.search({
        query,
        limit,
        collection,
        minScore,
        rerank: profile.rerank ?? true,
      });
      return results.map((r: HybridQueryResult) => r.file);
    }
  }
}

async function runWarmups(
  store: QMDStore,
  profile: BenchmarkProfile,
  query: BenchmarkQuery,
  warmupRuns: number,
  collection?: string,
): Promise<void> {
  if (warmupRuns <= 0) return;

  const limit = profile.limit ?? Math.max(query.expected_in_top_k, 10);
  for (let i = 0; i < warmupRuns; i++) {
    await runMode(store, profile, query.query, limit, collection);
  }
}

async function benchmarkQuery(
  store: QMDStore,
  profile: BenchmarkProfile,
  query: BenchmarkQuery,
  runs: number,
  warmupRuns: number,
  collection?: string,
): Promise<{ result: ComparisonQueryResult; allLatencies: number[] }> {
  await runWarmups(store, profile, query, warmupRuns, collection);

  const limit = profile.limit ?? Math.max(query.expected_in_top_k, 10);
  const measuredLatencies: number[] = [];
  const usageTrackers: ApiUsageTracker[] = [];
  let firstResultFiles: string[] = [];

  for (let i = 0; i < runs; i++) {
    const tracker = new ApiUsageTracker();
    const start = Date.now();
    const resultFiles = await withApiUsageTracking(tracker, async () =>
      runMode(store, profile, query.query, limit, collection),
    );
    const latency = Date.now() - start;

    measuredLatencies.push(latency);
    usageTrackers.push(tracker);

    if (i === 0) {
      firstResultFiles = resultFiles;
    }
  }

  const metrics = scoreResults(firstResultFiles, query.expected_files, query.expected_in_top_k);
  const usage = mergeUsageSummaries(...usageTrackers.map((tracker) => tracker.summarize()));

  return {
    result: {
      id: query.id,
      query: query.query,
      type: query.type,
      precision_at_k: metrics.precision_at_k,
      recall: metrics.recall,
      mrr: metrics.mrr,
      f1: metrics.f1,
      hits_at_k: metrics.hits_at_k,
      total_expected: query.expected_files.length,
      latency_ms: measuredLatencies.reduce((sum, value) => sum + value, 0) / measuredLatencies.length,
      latency_p95_ms: percentile(measuredLatencies, 0.95),
      top_files: firstResultFiles.slice(0, 10),
      usage,
    },
    allLatencies: measuredLatencies,
  };
}

function formatUsd(value: number): string {
  return `$${value.toFixed(value >= 0.1 ? 3 : 4)}`;
}

function formatSummaryTable(result: ComparisonBenchmarkResult): string {
  const lines: string[] = [];
  const pad = (value: string, width: number) => value.slice(0, width).padEnd(width);

  lines.push(
    [
      pad("Profile", 22),
      pad("Mode", 8),
      pad("P@k", 6),
      pad("Recall", 7),
      pad("MRR", 6),
      pad("F1", 6),
      pad("Avg ms", 8),
      pad("P95 ms", 8),
      pad("Search $", 10),
      pad("Prep $", 10),
    ].join(" ")
  );
  lines.push("-".repeat(105));

  for (const profile of result.profiles) {
    const summary = profile.summary;
    lines.push(
      [
        pad(profile.profile, 22),
        pad(profile.mode, 8),
        pad(summary.avg_precision.toFixed(2), 6),
        pad(summary.avg_recall.toFixed(2), 7),
        pad(summary.avg_mrr.toFixed(2), 6),
        pad(summary.avg_f1.toFixed(2), 6),
        pad(String(Math.round(summary.avg_latency_ms)), 8),
        pad(String(Math.round(summary.p95_latency_ms)), 8),
        pad(formatUsd(summary.total_search_cost_usd), 10),
        pad(formatUsd(summary.prepare_cost_usd), 10),
      ].join(" ")
    );
  }

  return lines.join("\n");
}

export async function runComparisonBenchmark(
  planPath: string,
  options: { json?: boolean; output?: string } = {},
): Promise<ComparisonBenchmarkResult> {
  const absolutePlanPath = resolve(planPath);
  const planDir = dirname(absolutePlanPath);
  const raw = readFileSync(absolutePlanPath, "utf-8");
  const plan = JSON.parse(raw) as ComparisonBenchmarkPlan;

  if (!Array.isArray(plan.profiles) || plan.profiles.length === 0) {
    throw new Error("Invalid comparison benchmark plan: missing 'profiles' array");
  }
  if (!Array.isArray(plan.queries) || plan.queries.length === 0) {
    throw new Error("Invalid comparison benchmark plan: missing 'queries' array");
  }

  const baseConfig = resolvePlanConfig(plan, absolutePlanPath);
  const dbRoot = resolve(planDir, plan.db_root ?? ".qmd-bench");
  mkdirSync(dbRoot, { recursive: true });

  const profileResults: ProfileComparisonResult[] = [];

  for (const profile of plan.profiles) {
    const config = deepClone(baseConfig);
    config.models = {
      ...(config.models ?? {}),
      ...(profile.models ?? {}),
    };

    const dbPath = profile.db_path
      ? resolveMaybeRelative(planDir, profile.db_path)
      : resolve(dbRoot, `${sanitizeProfileName(profile.name)}.sqlite`);

    const store = await createStore({
      dbPath,
      config,
    });

    const prepareUpdate = profile.prepare?.update ?? true;
    const prepareEmbed = profile.prepare?.embed ?? (profile.mode !== "bm25");
    const prepareForceEmbed = profile.prepare?.force_embed ?? false;

    const prepareTracker = new ApiUsageTracker();
    const prepareUpdateStart = Date.now();
    if (prepareUpdate) {
      await withApiUsageTracking(prepareTracker, async () => {
        await store.update();
      });
    }
    const updateMs = Date.now() - prepareUpdateStart;

    const prepareEmbedStart = Date.now();
    if (prepareEmbed) {
      await withApiUsageTracking(prepareTracker, async () => {
        await store.embed({ force: prepareForceEmbed });
      });
    }
    const embedMs = Date.now() - prepareEmbedStart;

    const prepareUsage = prepareTracker.summarize();

    const collection = profile.collection ?? plan.collection;
    const runs = profile.runs ?? plan.runs ?? 1;
    const warmupRuns = profile.warmup_runs ?? plan.warmup_runs ?? 0;
    const queryResults: ComparisonQueryResult[] = [];
    const allSearchLatencies: number[] = [];

    for (const query of plan.queries) {
      if (!options.json) {
        process.stderr.write(`  ${profile.name} / ${query.id}...`);
      }

      const { result, allLatencies } = await benchmarkQuery(store, profile, query, runs, warmupRuns, collection);
      queryResults.push(result);
      allSearchLatencies.push(...allLatencies);

      if (!options.json) {
        process.stderr.write(
          ` ${Math.round(result.latency_ms)}ms ${formatUsd(result.usage.estimated_cost_usd)}\n`,
        );
      }
    }

    await store.close();

    const queryCount = Math.max(queryResults.length, 1);
    const totalSearchCost = queryResults.reduce((sum, query) => sum + query.usage.estimated_cost_usd, 0);
    const totalApiCalls = queryResults.reduce((sum, query) => sum + query.usage.request_count, 0);
    const totalInputTokens = queryResults.reduce((sum, query) => sum + query.usage.input_tokens, 0);

    profileResults.push({
      profile: profile.name,
      description: profile.description,
      mode: profile.mode,
      collection,
      db_path: dbPath,
      prepare: {
        update_ms: updateMs,
        embed_ms: embedMs,
        usage: prepareUsage,
      },
      queries: queryResults,
      summary: {
        avg_precision: queryResults.reduce((sum, query) => sum + query.precision_at_k, 0) / queryCount,
        avg_recall: queryResults.reduce((sum, query) => sum + query.recall, 0) / queryCount,
        avg_mrr: queryResults.reduce((sum, query) => sum + query.mrr, 0) / queryCount,
        avg_f1: queryResults.reduce((sum, query) => sum + query.f1, 0) / queryCount,
        avg_latency_ms: queryResults.reduce((sum, query) => sum + query.latency_ms, 0) / queryCount,
        p95_latency_ms: percentile(allSearchLatencies, 0.95),
        total_search_cost_usd: totalSearchCost,
        avg_search_cost_usd: totalSearchCost / queryCount,
        prepare_cost_usd: prepareUsage.estimated_cost_usd,
        total_cost_usd: totalSearchCost + prepareUsage.estimated_cost_usd,
        total_api_calls: totalApiCalls + prepareUsage.request_count,
        total_input_tokens: totalInputTokens + prepareUsage.input_tokens,
      },
    });
  }

  const result: ComparisonBenchmarkResult = {
    timestamp: new Date().toISOString().replace(/[:.]/g, "").slice(0, 15),
    plan: absolutePlanPath,
    description: plan.description,
    profiles: profileResults,
  };

  if (options.output) {
    const outputPath = resolveMaybeRelative(planDir, options.output);
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, JSON.stringify(result, null, 2), "utf-8");
  }

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(`\nComparison benchmark: ${plan.description}\n`);
    console.log(formatSummaryTable(result));
    if (options.output) {
      console.log(`\nSaved JSON report to ${resolveMaybeRelative(planDir, options.output)}`);
    }
  }

  return result;
}
