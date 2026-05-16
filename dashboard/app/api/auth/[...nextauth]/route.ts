// NextAuth v5 route handlers — exposes the OAuth flow + callbacks at
//   /api/auth/signin
//   /api/auth/signout
//   /api/auth/callback/google
//   /api/auth/session
import { handlers } from "@/auth";

export const { GET, POST } = handlers;