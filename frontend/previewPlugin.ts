import type { Plugin } from 'vite'

/** Local-preview HTML diagnostics run before React, even if the app cannot render. */
export function previewPlugin(id: string, builtAt: string): Plugin {
  return {
    name: 'exp03-local-preview-identity',
    transformIndexHtml() {
      const cleanup = `
const previewCleanup = (async function () {
  const beforeController = !!navigator.serviceWorker?.controller;
  const registrations = navigator.serviceWorker ? await navigator.serviceWorker.getRegistrations() : [];
  const local = registrations.filter(r => new URL(r.scope).origin === location.origin);
  await Promise.all(local.map(r => r.unregister()));
  const names = 'caches' in window ? await caches.keys() : [];
  await Promise.all(names.map(name => caches.delete(name)));
  const result = { controllerBefore: beforeController, registrationsRemoved: local.length, cachesDeleted: names.length, cacheNames: names };
  window.__EXP03_PREVIEW_CACHE_AUDIT__ = result;
  const key = 'exp03-preview-controller-reset:' + ${JSON.stringify(id)};
  if (beforeController && !sessionStorage.getItem(key)) {
    sessionStorage.setItem(key, '1');
    const url = new URL(location.href); url.searchParams.set('preview-build', ${JSON.stringify(id)});
    location.replace(url.href);
  }
  return result;
})();
window.__EXP03_PREVIEW_CLEANUP__ = new Promise(resolve => {
  const timer = setTimeout(() => { const result = { error: 'Cache cleanup timeout (2 s)' }; window.__EXP03_PREVIEW_CACHE_AUDIT__ = result; resolve(result); }, 2000);
  previewCleanup.then(result => { clearTimeout(timer); resolve(result); }, error => {
    clearTimeout(timer); const result = { error: String(error) }; window.__EXP03_PREVIEW_CACHE_AUDIT__ = result; resolve(result);
  });
});
`
      return [
        { tag: 'meta', attrs: { name: 'collembola-preview-build', content: id }, injectTo: 'head-prepend' },
        { tag: 'meta', attrs: { name: 'collembola-preview-built-at', content: builtAt }, injectTo: 'head-prepend' },
        { tag: 'script', attrs: { 'data-preview-cache-cleanup': id }, children: cleanup, injectTo: 'head-prepend' },
        { tag: 'div', attrs: { id: 'preview-build-badge', role: 'note', 'data-build-id': id, 'aria-label': 'Local preview build',
          style: 'position:fixed;right:10px;bottom:8px;z-index:99999;max-width:490px;background:#172b45;color:#fff;border:2px solid #a78bfa;border-radius:6px;padding:8px 10px;font:11px/1.5 monospace;overflow-wrap:anywhere;box-shadow:0 2px 10px #0006' },
          children: `<strong>LOCAL PREVIEW · ${id}</strong><br>Build UTC: ${builtAt}<br><span id="preview-js-build">JS: čekam novi bundle…</span><br><span id="preview-api-build">API: čekam potvrdu…</span><br><span id="preview-cache-status">Cache: provjera…</span>`,
          injectTo: 'body-prepend' },
      ]
    },
  }
}
