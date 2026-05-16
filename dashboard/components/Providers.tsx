"use client";

// Client-side context providers (NextAuth's SessionProvider is the only
// one for now, but keeping the wrapper makes adding more straightforward).
import { SessionProvider } from "next-auth/react";

export default function Providers({ children }: { children: React.ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>;
}