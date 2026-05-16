"use client";

// Subscribe-to-alerts button. Two states:
//   1. Signed out → "Sign in with Google to subscribe" (triggers OAuth flow)
//   2. Signed in  → POST /api/subscribe with the authenticated email
//
// No password fields, no manual email entry — email comes from the session.
import { useState } from "react";
import { signIn, useSession } from "next-auth/react";

type Status = "idle" | "loading" | "subscribed" | "error";

export default function SubscribeButton() {
  const { data: session, status } = useSession();
  const [state, setState] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");

  if (status === "loading") {
    return (
      <button disabled className="rounded border border-bg-ring bg-bg-card px-3 py-1.5 font-mono text-xs text-ink-dim">
        Loading…
      </button>
    );
  }

  if (!session?.user?.email) {
    return (
      <button
        onClick={() => signIn("google", { redirectTo: "/" })}
        className="inline-flex items-center gap-2 rounded border border-accent-lblue bg-accent-dblue/30 px-3 py-1.5
                   font-mono text-xs text-accent-lblue hover:bg-accent-dblue/50"
      >
        🔔 Sign in to subscribe
      </button>
    );
  }

  if (state === "subscribed") {
    return (
      <span className="inline-flex items-center gap-2 rounded border border-green-700 bg-green-900/30 px-3 py-1.5 font-mono text-xs text-green-400">
        ✓ Subscribed as {session.user.email}
      </span>
    );
  }

  async function handleSubscribe() {
    setState("loading");
    setErrorMsg("");
    try {
      const res = await fetch("/api/subscribe", { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      setState("subscribed");
    } catch (e) {
      setState("error");
      setErrorMsg(e instanceof Error ? e.message : "Subscribe failed");
    }
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        onClick={handleSubscribe}
        disabled={state === "loading"}
        className="inline-flex items-center gap-2 rounded border border-accent-lblue bg-accent-dblue/30 px-3 py-1.5
                   font-mono text-xs text-accent-lblue hover:bg-accent-dblue/50 disabled:opacity-50"
      >
        {state === "loading" ? "Subscribing…" : `🔔 Subscribe ${session.user.email}`}
      </button>
      {state === "error" && (
        <span className="font-mono text-[10px] text-accent-red">{errorMsg}</span>
      )}
    </div>
  );
}