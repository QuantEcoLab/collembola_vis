import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const stripBase = (path: string) => path.replace(/^\/collembola/, '')

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/',
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
