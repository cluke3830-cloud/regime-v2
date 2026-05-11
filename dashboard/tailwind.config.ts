import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Bloomberg-dark palette
        bg: {
          DEFAULT: "#0a0e14",
          panel: "#0f1521",
          card:  "#121a28",
          ring:  "#1e2a3a",
        },
        accent: {
          dblue:  "#1F3864",
          lblue:  "#2E75B6",
          amber:  "#f5a623",
          green:  "#22c55e",
          lime:   "#84cc16",
          grey:   "#a3a3a3",
          orange: "#f97316",
          red:    "#ef4444",
        },
        ink: {
          DEFAULT: "#e6edf3",
          muted:   "#9aa6b2",
          dim:     "#5c6c80",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
