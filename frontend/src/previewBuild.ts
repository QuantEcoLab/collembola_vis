declare const __EXP03_LOCAL_PREVIEW__: boolean
declare const __EXP03_BUILD_ID__: string
declare const __EXP03_BUILT_AT__: string

type CacheAudit = { controllerBefore?: boolean; registrationsRemoved?: number; cachesDeleted?: number; error?: string }
declare global {
  interface Window {
    __EXP03_PREVIEW_CLEANUP__?: Promise<CacheAudit>
    __EXP03_PREVIEW_CACHE_AUDIT__?: CacheAudit
  }
}

export const localPreview = typeof __EXP03_LOCAL_PREVIEW__ !== 'undefined' && __EXP03_LOCAL_PREVIEW__

export function announcePreviewBuild() {
  if (!localPreview) return
  const set = (id: string, text: string) => { const node = document.getElementById(id); if (node) node.textContent = text }
  const htmlBuild = document.querySelector<HTMLMetaElement>('meta[name="collembola-preview-build"]')?.content
  set('preview-js-build', `JS: ${__EXP03_BUILD_ID__} · ${htmlBuild === __EXP03_BUILD_ID__ ? 'HTML/JS OK' : 'HTML/JS NE ODGOVARAJU'}`)
  const cache = window.__EXP03_PREVIEW_CACHE_AUDIT__
  set('preview-cache-status', cache?.error ? `Cache cleanup: ${cache.error}` : `SW uklonjen: ${cache?.registrationsRemoved ?? 0}; cache uklonjen: ${cache?.cachesDeleted ?? 0}; localStorage sačuvan`)
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 3000)
  fetch(`/api/preview-build?client-build=${encodeURIComponent(__EXP03_BUILD_ID__)}`, { cache: 'no-store', signal: controller.signal })
    .then(async response => {
      if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error(`HTTP ${response.status}: pogrešan backend/bundle`)
      return response.json()
    })
    .then(data => {
      set('preview-api-build', `API: ${data.build_id} · PID ${data.pid} · ${data.build_id === __EXP03_BUILD_ID__ ? 'JS/API OK' : 'JS/API NE ODGOVARAJU'}`)
      const badge = document.getElementById('preview-build-badge')
      if (badge) { badge.dataset.jsBuild = __EXP03_BUILD_ID__; badge.dataset.apiBuild = data.build_id; badge.dataset.builtAt = __EXP03_BUILT_AT__ }
    })
    .catch(error => set('preview-api-build', `API provjera nije uspjela: ${error.message}`))
    .finally(() => clearTimeout(timer))
}
