import fs from "fs/promises";
import path from "path";
import type { AssetIndex, AssetPayload, SummaryPayload } from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

export async function loadAssetIndex(): Promise<AssetIndex> {
  const raw = await fs.readFile(path.join(DATA_DIR, "assets.json"), "utf8");
  return JSON.parse(raw) as AssetIndex;
}

export async function loadSummary(): Promise<SummaryPayload> {
  const raw = await fs.readFile(path.join(DATA_DIR, "summary.json"), "utf8");
  return JSON.parse(raw) as SummaryPayload;
}

export async function loadAsset(safeTicker: string): Promise<AssetPayload> {
  const file = path.join(DATA_DIR, "regimes", `${safeTicker}.json`);
  const raw = await fs.readFile(file, "utf8");
  return JSON.parse(raw) as AssetPayload;
}
