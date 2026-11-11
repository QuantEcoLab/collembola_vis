import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App'
import { announcePreviewBuild, localPreview } from './previewBuild'

function renderApp() {
  announcePreviewBuild()
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, '') || '/'}>
        <App />
      </BrowserRouter>
    </StrictMode>,
  )
}

if (localPreview && window.__EXP03_PREVIEW_CLEANUP__) {
  void window.__EXP03_PREVIEW_CLEANUP__.then(renderApp)
} else renderApp()
