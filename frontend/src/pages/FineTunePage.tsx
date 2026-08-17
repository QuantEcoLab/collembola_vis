import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { LogOut, RefreshCw } from 'lucide-react'
import FineTunePanel from '../components/FineTunePanel'
import { listModels } from '../api/client'
import { useAuthStore } from '../store/authStore'
import type { ModelInfo } from '../api/types'

export default function FineTunePage() {
  const navigate = useNavigate()
  const { logout, role } = useAuthStore()
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loadingModels, setLoadingModels] = useState(true)

  useEffect(() => {
    if (role !== 'admin') {
      navigate('/projects', { replace: true })
    }
  }, [role])

  const fetchModels = () => {
    setLoadingModels(true)
    listModels()
      .then(setModels)
      .catch(() => {})
      .finally(() => setLoadingModels(false))
  }

  useEffect(() => { fetchModels() }, [])

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      {/* Header */}
      <header className="shrink-0 bg-white border-b px-5 h-12 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <span className="font-semibold text-gray-900">Collembola</span>
          <nav className="flex items-center gap-1">
            <Link
              to="/projects"
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Projects
            </Link>
            <Link
              to="/workspace"
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Workspace
            </Link>
            <Link
              to="/finetune"
              className="text-sm px-3 py-1 rounded-md bg-gray-100 text-gray-900 font-medium"
            >
              Fine-Tune
            </Link>
          </nav>
        </div>
        <button
          onClick={() => { logout(); navigate('/login', { replace: true }) }}
          className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700"
        >
          <LogOut size={14} />
          Sign out
        </button>
      </header>

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-8 space-y-8">

          {/* Fine-tune panel */}
          <section>
            <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
              Train on user corrections
            </h2>
            <div className="bg-white rounded-xl border p-5">
              <FineTunePanel onModelSelected={fetchModels} />
            </div>
          </section>

          {/* Models list */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                Available models
              </h2>
              <button
                onClick={fetchModels}
                className="text-gray-400 hover:text-gray-600 transition-colors"
                title="Refresh"
              >
                <RefreshCw size={13} className={loadingModels ? 'animate-spin' : ''} />
              </button>
            </div>
            <div className="bg-white rounded-xl border divide-y">
              {models.length === 0 ? (
                <p className="px-4 py-3 text-sm text-gray-400">No models found</p>
              ) : (
                models.map((m) => (
                  <div key={m.path} className="flex items-center justify-between px-4 py-3">
                    <div>
                      <p className="text-sm font-medium text-gray-900">{m.name}</p>
                      <p className="text-xs text-gray-400 font-mono">{m.path}</p>
                    </div>
                    <div className="text-right shrink-0 ml-4">
                      <p className="text-xs text-gray-500">{m.size_mb} MB</p>
                      <p className="text-xs text-gray-400">
                        {new Date(m.mtime).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

        </div>
      </div>
    </div>
  )
}
