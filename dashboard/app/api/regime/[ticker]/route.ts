import { spawn } from "child_process";
import fs from "fs/promises";
import { stat } from "fs/promises";
import path from "path";
import { NextRequest } from "next/server";

// Cache fresh for 24 hours — recompute once per trading day
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

// Only valid US stock ticker chars (no path traversal possible)
const TICKER_RE = /^[A-Z0-9.\-^]{1,20}$/;

function safeName(ticker: string): string {
  return ticker.replace(/=/g, "_").replace(/\//g, "_");
}

async function isFileFresh(filePath: string): Promise<boolean> {
  try {
    const s = await stat(filePath);
    return Date.now() - s.mtimeMs < CACHE_TTL_MS;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Mode A — proxy to EC2 Python server (used in production on Vercel)
// Set REGIME_API_URL=http://<ec2-public-ip>:8051 in Vercel env vars.
// ---------------------------------------------------------------------------
async function fetchFromApiServer(ticker: string): Promise<Response> {
  const baseUrl = process.env.REGIME_API_URL!.replace(/\/$/, "");
  const url = `${baseUrl}/regime/${encodeURIComponent(ticker)}`;
  console.log(`[regime-api] proxying to ${url}`);

  const upstream = await fetch(url, {
    // Give the Python server up to 120s to compute
    signal: AbortSignal.timeout(120_000),
  });

  const body = await upstream.text();
  if (!upstream.ok) {
    return Response.json(
      { error: `EC2 server returned ${upstream.status}: ${body}` },
      { status: upstream.status }
    );
  }
  return new Response(body, {
    headers: { "Content-Type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Mode B — spawn local Python subprocess (local next dev only)
// ---------------------------------------------------------------------------
function runComputeScript(ticker: string, outDir: string): Promise<void> {
  const scriptPath = path.join(process.cwd(), "..", "scripts", "compute_regime.py");
  const backend = process.env.REGIME_BACKEND ?? "yfinance";

  return new Promise((resolve, reject) => {
    const proc = spawn(
      "python3",
      [scriptPath, ticker, "--out-dir", outDir, "--backend", backend],
      { env: { ...process.env } }
    );

    let stderr = "";
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.stdout.on("data", (d: Buffer) => { process.stdout.write(d); });

    const timer = setTimeout(() => {
      proc.kill();
      reject(new Error("compute_regime.py timed out after 120s"));
    }, 120_000);

    proc.on("close", (code: number | null) => {
      clearTimeout(timer);
      if (code === 0) resolve();
      else reject(new Error(`compute_regime.py exited ${code}: ${stderr}`));
    });

    proc.on("error", (e: Error) => {
      clearTimeout(timer);
      reject(e);
    });
  });
}

async function fetchFromSubprocess(ticker: string): Promise<Response> {
  const safe = safeName(ticker);
  const regimesDir = path.join(process.cwd(), "public", "data", "regimes");
  const filePath = path.join(regimesDir, `${safe}.json`);

  if (!(await isFileFresh(filePath))) {
    console.log(`[regime-api] spawning compute_regime.py for ${ticker}`);
    await runComputeScript(ticker, regimesDir);
  } else {
    console.log(`[regime-api] file cache hit: ${ticker}`);
  }

  const rawJson = await fs.readFile(filePath, "utf8");
  return new Response(rawJson, {
    headers: { "Content-Type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------
export async function GET(
  _req: NextRequest,
  { params }: { params: { ticker: string } }
) {
  const raw = params.ticker.toUpperCase().trim();

  if (!TICKER_RE.test(raw)) {
    return Response.json({ error: "Invalid ticker symbol" }, { status: 400 });
  }

  try {
    if (process.env.REGIME_API_URL) {
      // Production: proxy to EC2 Python server
      return await fetchFromApiServer(raw);
    } else {
      // Local dev: spawn compute_regime.py subprocess
      return await fetchFromSubprocess(raw);
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[regime-api] ${raw} failed:`, msg);
    return Response.json(
      { error: `Failed to compute regime for ${raw}: ${msg}` },
      { status: 500 }
    );
  }
}