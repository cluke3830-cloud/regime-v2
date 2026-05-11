import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Regime_v2 — Multi-Asset Live Regime",
  description:
    "Live 5-regime classification across 10 assets — TVTP-MSAR champion, rule baseline, equity curves, transition matrices.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen relative">{children}</body>
    </html>
  );
}
