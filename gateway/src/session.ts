import type { RequestHandler } from "express";

export const SESSION_COOKIE = "ng_session";

/**
 * A signed cookie carries a `user_id` from a small set of seeded demo Profiles.
 * No password, no sign-up, no email address.
 */
export function session(demoUserId: string): RequestHandler {
  return (req, res, next) => {
    const signed = req.signedCookies?.[SESSION_COOKIE];
    const userId = typeof signed === "string" && signed.length > 0 ? signed : demoUserId;
    if (userId !== signed) {
      res.cookie(SESSION_COOKIE, userId, {
        httpOnly: true,
        signed: true,
        sameSite: "lax",
        path: "/",
      });
    }
    res.locals.userId = userId;
    next();
  };
}
