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

function runComputeScript(ticker: string, outDir: string): Promise<void> {
  // process.cwd() in Next.js = dashboard/ directory
  const scriptPath = path.join(process.cwd(), "..", "scripts", "compute_regime.py");

  // REGIME_BACKEND=ibkr routes through live IB Gateway on EC2 (port 4004).
  // Default is yfinance (cached, works without IBKR connectivity).
  const backend = process.env.REGIME_BACKEND ?? "yfinance";

  return new Promise((resolve, reject) => {
    const proc = spawn(
      "python3",
      [scriptPath, ticker, "--out-dir", outDir, "--backend", backend],
      { env: { ...process.env } },
    );

    let stderr = "";
    proc.stderr.on("data", (d: Buffer) => {
      stderr += d.toString();
    });
    // Print progress lines from the script to server console
    proc.stdout.on("data", (d: Buffer) => {
      process.stdout.write(d);
    });

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

export async function GET(
  _req: NextRequest,
  { params }: { params: { ticker: string } }
) {
  const raw = params.ticker.toUpperCase().trim();

  if (!TICKER_RE.test(raw)) {
    return Response.json({ error: "Invalid ticker symbol" }, { status: 400 });
  }

  const safe = safeName(raw);
  const regimesDir = path.join(process.cwd(), "public", "data", "regimes");
  const filePath = path.join(regimesDir, `${safe}.json`);

  try {
    if (!(await isFileFresh(filePath))) {
      console.log(`[regime-api] computing ${raw}…`);
      await runComputeScript(raw, regimesDir);
      console.log(`[regime-api] ${raw} done`);
    } else {
      console.log(`[regime-api] cache hit: ${raw}`);
    }

    const rawJson = await fs.readFile(filePath, "utf8");
    return new Response(rawJson, {
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[regime-api] ${raw} failed:`, msg);
    return Response.json(
      { error: `Failed to compute regime for ${raw}: ${msg}` },
      { status: 500 }
    );
  }
}