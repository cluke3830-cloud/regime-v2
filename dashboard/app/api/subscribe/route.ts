// Auth-gated subscribe / unsubscribe endpoint.
//
// POST  /api/subscribe   — subscribes the signed-in user's email
// DELETE /api/subscribe  — unsubscribes the signed-in user's email
//
// The browser never sees the EC2 backend URL or the shared REGIME_API_TOKEN;
// both live as server-side env vars on Vercel. This route is the only path
// from public client → backend, and it verifies the NextAuth session first.
//
// Required env vars (set in Vercel project settings):
//   REGIME_API_URL    — e.g. http://ec2-3-237-3-9.compute-1.amazonaws.com:8051
//   REGIME_API_TOKEN  — random shared secret; backend rejects requests without it
import { NextResponse } from "next/server";
import { auth } from "@/auth";

const BACKEND_URL = process.env.REGIME_API_URL;
const BACKEND_TOKEN = process.env.REGIME_API_TOKEN;

async function callBackend(method: "POST" | "DELETE", email: string) {
  if (!BACKEND_URL) {
    throw new Error("REGIME_API_URL not configured");
  }
  const res = await fetch(`${BACKEND_URL}/subscribe`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(BACKEND_TOKEN ? { Authorization: `Bearer ${BACKEND_TOKEN}` } : {}),
    },
    body: JSON.stringify({ email }),
    cache: "no-store",
  });
  return res;
}

export async function POST() {
  const session = await auth();
  if (!session?.user?.email) {
    return NextResponse.json({ error: "not signed in" }, { status: 401 });
  }

  try {
    const res = await callBackend("POST", session.user.email);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(body, { status: res.status });
    }
    return NextResponse.json({ status: "subscribed", email: session.user.email });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "backend unreachable" },
      { status: 502 },
    );
  }
}

export async function DELETE() {
  const session = await auth();
  if (!session?.user?.email) {
    return NextResponse.json({ error: "not signed in" }, { status: 401 });
  }
  try {
    const res = await callBackend("DELETE", session.user.email);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(body, { status: res.status });
    }
    return NextResponse.json({ status: "unsubscribed", email: session.user.email });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "backend unreachable" },
      { status: 502 },
    );
  }
}
