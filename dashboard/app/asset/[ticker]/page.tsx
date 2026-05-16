import { loadAsset, loadAssetIndex, loadSummary } from "@/lib/data";
import TopBar from "@/components/TopBar";
import AssetDetail from "@/components/AssetDetail";
import DynamicAssetLoader from "./DynamicAssetLoader";

// Pre-build the fixed 10-asset universe at build time; any other ticker is
// rendered on-demand via the /api/regime/[ticker] compute pipeline.
export const dynamicParams = true;

export async function generateStaticParams() {
  const index = await loadAssetIndex();
  return index.universe.map((u) => ({ ticker: u.safe }));
}

export default async function AssetPage({
  params,
}: {
  params: { ticker: string };
}) {
  const [index, summary] = await Promise.all([loadAssetIndex(), loadSummary()]);

  // Attempt to serve from pre-built static JSON; fall back to client-side
  // on-demand compute for any ticker not in the universe.
  let asset = null;
  try {
    asset = await loadAsset(params.ticker);
  } catch {
    // File not found — dynamic ticker, handled below
  }

  return (
    <main className="min-h-screen">
      <TopBar universe={index.universe} generatedAt={summary.generated_at} />
      {asset ? (
        <AssetDetail asset={asset} modelRunAt={summary.generated_at} />
      ) : (
        <DynamicAssetLoader ticker={params.ticker.toUpperCase()} />
      )}
    </main>
  );
}