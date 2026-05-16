import type { Metadata } from "next";
import "./globals.css";
import Providers from "@/components/Providers";

export const metadata: Metadata = {
  title: "Regime_v2 — Multi-Asset Live Regime",
  description:
    "Live regime classification across 15 assets (equity, FX, crypto) — TVTP-MSAR champion, market consensus, alert subscriptions.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen relative">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
