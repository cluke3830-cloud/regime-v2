# Auth + Email Setup Guide (Phase 9)

Three one-time setups before the dashboard can sign users in with Google
and send them regime alerts. None require user passwords anywhere in the UX.

---

## 1. Resend (outbound email) — 3 minutes

1. Sign up at https://resend.com (free tier: 3,000 emails/month, 100/day).
2. **Onboarding → Add API Key** → name it `regime-monitor` → copy the
   `re_...` key. Store it; the dashboard shows it once.
3. (Optional, for production) Add and verify a domain. Until then, you can
   send from the default `onboarding@resend.dev` for testing.

Set on EC2 (where `send_regime_alerts.py` runs):

```bash
export RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxxxxx
export ALERT_FROM_EMAIL="Regime Monitor <onboarding@resend.dev>"   # or your verified domain
```

Test:

```bash
cd ~/regime-v2  # or wherever the repo lives on EC2
python scripts/send_regime_alerts.py --dry-run    # prints what would be sent
python scripts/send_regime_alerts.py              # actually sends
```

---

## 2. Google OAuth (sign-in for users) — 5 minutes

1. Open https://console.cloud.google.com/apis/credentials
2. Top bar → **Select project** → **New project** → name it
   `regime-monitor` → Create.
3. Sidebar → **OAuth consent screen**:
   - User type: **External** → Create
   - App name: `Regime Monitor`
   - User support email: your email
   - Developer contact: your email
   - Save & continue (skip scopes, test users)
4. Sidebar → **Credentials** → **Create credentials** → **OAuth client ID**:
   - Application type: **Web application**
   - Name: `regime-monitor-web`
   - **Authorized redirect URIs** (add both):
     ```
     http://localhost:3000/api/auth/callback/google
     https://<your-vercel-domain>.vercel.app/api/auth/callback/google
     ```
   - Create → copy the **Client ID** (`*.apps.googleusercontent.com`) and
     **Client Secret**.

Generate an `AUTH_SECRET`:

```bash
openssl rand -base64 32
```

---

## 3. Vercel environment variables

In Vercel project settings → **Environment Variables**, add:

| Name                | Value                                        | Where used               |
| ------------------- | -------------------------------------------- | ------------------------ |
| `AUTH_SECRET`       | `<from openssl above>`                       | NextAuth signs JWTs       |
| `AUTH_GOOGLE_ID`    | `<client id>.apps.googleusercontent.com`     | Google OAuth flow        |
| `AUTH_GOOGLE_SECRET`| `<client secret>`                            | Google OAuth flow        |
| `AUTH_TRUST_HOST`   | `true`                                       | Required on Vercel        |
| `REGIME_API_URL`    | `http://<ec2-host>:8051`                     | /api/subscribe proxy     |
| `REGIME_API_TOKEN`  | `<random 32+ char string>`                   | Shared secret backend    |

Set the same `REGIME_API_TOKEN` on EC2:

```bash
export REGIME_API_TOKEN="<same string>"
# restart the API server so it picks up the new env var
```

---

## How the flow works end-to-end

```
User → Vercel dashboard (Next.js)
            │ click "Sign in"
            ▼
       Google OAuth ─→ user picks account ─→ NextAuth callback
            │
            ▼
      Session JWT (cookie, server-side validation)
            │ click "Subscribe alerts"
            ▼
   /api/subscribe (Next.js route)
            │ verifies session, attaches REGIME_API_TOKEN
            ▼
   POST /subscribe on EC2 Flask backend
            │ verifies token, writes data/subscribers.json
            ▼
       (next regime change)
            │
   send_regime_alerts.py picks up subscriber
            │
            ▼
   Resend HTTP API → user's inbox
```

User never sees:
- the Google OAuth client secret
- the REGIME_API_TOKEN
- the RESEND_API_KEY
- the EC2 backend URL

User does see:
- Sign-in-with-Google button
- Their own email (pulled from the Google session)
- "Subscribe" button
- Alert emails in their inbox
