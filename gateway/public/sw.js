// Shell cache only. `/api` is never intercepted, so there is no offline
// logging by construction, not just by intention.
const CACHE = "ng-shell-v1";
const SHELL = [
  "/",
  "/style.css",
  "/client.js",
  "/manifest.webmanifest",
  "/icons/icon.svg",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.pathname.startsWith("/api/")) return;

  // Network first so a deploy is picked up on the next open; cache is the
  // fallback that keeps the shell rendering with no signal.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() =>
        caches
          .match(event.request)
          .then((hit) => hit ?? (event.request.mode === "navigate" ? caches.match("/") : undefined))
          .then((hit) => hit ?? Response.error()),
      ),
  );
});
