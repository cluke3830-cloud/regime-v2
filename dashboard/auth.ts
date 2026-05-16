// NextAuth v5 (Auth.js) configuration — Google OAuth only.
//
// Required env vars (set in .env.local for dev, Vercel project settings for prod):
//   AUTH_SECRET            — random 32-byte string (npx auth secret)
//   AUTH_GOOGLE_ID         — OAuth client ID from Google Cloud Console
//   AUTH_GOOGLE_SECRET     — OAuth client secret from Google Cloud Console
//   AUTH_TRUST_HOST=true   — required on Vercel preview/prod deployments
//
// Google Cloud Console — Authorized redirect URIs:
//   http://localhost:3000/api/auth/callback/google         (dev)
//   https://<your-domain>.vercel.app/api/auth/callback/google  (prod)
import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    Google({
      clientId: process.env.AUTH_GOOGLE_ID,
      clientSecret: process.env.AUTH_GOOGLE_SECRET,
    }),
  ],
  // JWT sessions — stateless, no database needed for v1.
  // Email is the only field we expose to the client; everything else lives
  // server-side via auth().
  session: { strategy: "jwt" },
  callbacks: {
    async session({ session, token }) {
      if (session.user && token.sub) session.user.id = token.sub;
      return session;
    },
  },
  pages: {
    // Use the default NextAuth sign-in page (the Google button does the work)
  },
});