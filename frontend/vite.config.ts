import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { previewPlugin } from './previewPlugin'

const stripBase = (path: string) => path.replace(/^\/collembola/, '')
const localPreview = process.env.VITE_LOCAL_PREVIEW === '1'
const buildId = process.env.VITE_PREVIEW_BUILD_ID ?? ''
const builtAt = process.env.VITE_PREVIEW_BUILT_AT ?? ''

export default defineConfig({
  plugins: [react(), tailwindcss(), ...(localPreview ? [previewPlugin(buildId, builtAt)] : [])],
  define: {
    __EXP03_LOCAL_PREVIEW__: JSON.stringify(localPreview),
    __EXP03_BUILD_ID__: JSON.stringify(buildId),
    __EXP03_BUILT_AT__: JSON.stringify(builtAt),
  },
  build: localPreview ? {
    rollupOptions: { output: {
      entryFileNames: `assets/${buildId}/[name]-[hash].js`,
      chunkFileNames: `assets/${buildId}/[name]-[hash].js`,
      assetFileNames: `assets/${buildId}/[name]-[hash][extname]`,
    } },
  } : undefined,
  base: process.env.VITE_BASE_PATH ?? '/',
  server: {
    port: 9173,
    proxy: {
      // Jana's /collembola/* routes let the dev server work against the
      // advandeb.com/collembola base path; retargeted to 9000 to match prod.
      '/collembola/api': {
        target: 'http://127.0.0.1:9000',
        changeOrigin: true,
        rewrite: stripBase,
      },
      '/collembola/ws': {
        target: 'ws://127.0.0.1:9000',
        ws: true,
        rewrite: stripBase,
      },
      '/collembola/files': {
        target: 'http://127.0.0.1:9000',
        changeOrigin: true,
      },
      '/api': 'http://127.0.0.1:9000',
      '/ws': {
        target: 'ws://127.0.0.1:9000',
        ws: true,
      },
      '/files': 'http://127.0.0.1:9000',
    },
  },
})
