/**
 * Types for the QMD benchmark harness.
 *
 * A benchmark fixture defines queries with expected results.
 * The harness runs each query through multiple search backends
 * and measures precision, recall, MRR, and latency.
 */

export interface BenchmarkQuery {
  /** Unique identifier for the query */
  id: string;
  /** The search query text */
  query: string;
  /** Query difficulty/type for grouping results */
  type: "exact" | "semantic" | "topical" | "cross-domain" | "alias";
  /** Human-readable description of what this tests */
  description: string;
  /** File paths (relative to collection) that should appear in results */
  expected_files: string[];
  /** How many of expected_files should appear in top-k results */
  expected_in_top_k: number;
}

export interface BenchmarkFixture {
  /** Description of the benchmark */
  description: string;
  /** Fixture format version */
  version: number;
  /** Optional collection to search within */
  collection?: string;
  /** The test queries */
  queries: BenchmarkQuery[];
}

export interface BackendResult {
  /** Fraction of top-k results that are relevant */
  precision_at_k: number;
  /** Fraction of expected files found anywhere in results */
  recall: number;
  /** Reciprocal rank of first relevant result (1/rank, 0 if not found) */
  mrr: number;
  /** Harmonic mean of precision_at_k and recall */
  f1: number;
  /** Number of expected files found in top-k */
  hits_at_k: number;
  /** Total expected files */
  total_expected: number;
  /** Wall-clock latency in milliseconds */
  latency_ms: number;
  /** Top result file paths (for inspection) */
  top_files: string[];
}

export interface QueryResult {
  id: string;
  query: string;
  type: string;
  backends: Record<string, BackendResult>;
}

export interface BenchmarkResult {
  timestamp: string;
  fixture: string;
  results: QueryResult[];
  summary: Record<string, {
    avg_precision: number;
    avg_recall: number;
    avg_mrr: number;
    avg_f1: number;
    avg_latency_ms: number;
  }>;
}

// =============================================================================
// Profile-based comparison benchmark
// =============================================================================

export type BenchmarkMode = "bm25" | "vector" | "hybrid" | "full";

export interface BenchmarkProfile {
  /** Stable profile id used in summaries */
  name: string;
  /** Human-readable profile description */
  description?: string;
  /** Search pipeline to benchmark */
  mode: BenchmarkMode;
  /** Optional per-profile collection override */
  collection?: string;
  /** Optional search result limit override */
  limit?: number;
  /** Optional min-score override */
  min_score?: number;
  /** Search rerank override for hybrid/full modes */
  rerank?: boolean;
  /** Model overrides for this profile */
  models?: {
    embed?: string;
    rerank?: string;
    generate?: string;
  };
  /** Optional explicit DB path for this profile */
  db_path?: string;
  /** Number of measured runs per query */
  runs?: number;
  /** Warm-up runs before timing starts */
  warmup_runs?: number;
  /** Preparation steps for this profile */
  prepare?: {
    update?: boolean;
    embed?: boolean;
    force_embed?: boolean;
  };
}

export interface ComparisonBenchmarkPlan {
  /** Description of the comparison benchmark */
  description: string;
  /** Fixture format version */
  version: number;
  /** Optional DB directory for generated per-profile databases */
  db_root?: string;
  /** Optional default collection override */
  collection?: string;
  /** Optional base config override. If omitted, current qmd YAML config is used. */
  config?: {
    global_context?: string;
    editor_uri?: string;
    editor_uri_template?: string;
    collections: Record<string, {
      path: string;
      pattern: string;
      ignore?: string[];
      context?: Record<string, string>;
      update?: string;
      includeByDefault?: boolean;
    }>;
    models?: {
      embed?: string;
      rerank?: string;
      generate?: string;
    };
  };
  /** Default measured runs per query */
  runs?: number;
  /** Default warm-up runs per query */
  warmup_runs?: number;
  /** Profiles to compare */
  profiles: BenchmarkProfile[];
  /** Benchmark queries */
  queries: BenchmarkQuery[];
}

export interface ApiUsageSummary {
  request_count: number;
  input_tokens: number;
  estimated_cost_usd: number;
  by_model: Record<string, {
    request_count: number;
    input_tokens: number;
    estimated_cost_usd: number;
  }>;
}

export interface ComparisonQueryResult {
  id: string;
  query: string;
  type: string;
  precision_at_k: number;
  recall: number;
  mrr: number;
  f1: number;
  hits_at_k: number;
  total_expected: number;
  latency_ms: number;
  latency_p95_ms: number;
  top_files: string[];
  usage: ApiUsageSummary;
}

export interface ProfileComparisonResult {
  profile: string;
  description?: string;
  mode: BenchmarkMode;
  collection?: string;
  db_path: string;
  prepare: {
    update_ms: number;
    embed_ms: number;
    usage: ApiUsageSummary;
  };
  queries: ComparisonQueryResult[];
  summary: {
    avg_precision: number;
    avg_recall: number;
    avg_mrr: number;
    avg_f1: number;
    avg_latency_ms: number;
    p95_latency_ms: number;
    total_search_cost_usd: number;
    avg_search_cost_usd: number;
    prepare_cost_usd: number;
    total_cost_usd: number;
    total_api_calls: number;
    total_input_tokens: number;
  };
}

export interface ComparisonBenchmarkResult {
  timestamp: string;
  plan: string;
  description: string;
  profiles: ProfileComparisonResult[];
}
