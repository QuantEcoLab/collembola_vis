import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogOut, Plus, FolderOpen } from 'lucide-react'
import { listProjects, createProject } from '../api/client'
import { thumbnailUrl } from '../api/client'
import { useAuthStore } from '../store/authStore'
import type { Project } from '../api/types'

export default function ProjectsPage() {
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)

  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [formError, setFormError] = useState('')

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    setFormError('')
    try {
      const res = await createProject(newName.trim(), newDesc.trim())
      navigate(`/projects/${res.id}`)
    } catch (err: any) {
      setFormError(err.message || 'Failed to create project')
    } finally {
      setCreating(false)
    }
  }

  const fmtDate = (iso: string) => {
    try { return new Date(iso).toLocaleDateString() } catch { return iso }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      {/* Header */}
      <header className="shrink-0 bg-white border-b px-5 h-12 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <span className="font-semibold text-gray-900">Collembola</span>
          <nav className="flex items-center gap-1">
            <Link
              to="/projects"
              className="text-sm px-3 py-1 rounded-md bg-gray-100 text-gray-900 font-medium transition-colors"
            >
              Projects
            </Link>
            <Link
              to="/workspace"
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Workspace
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

      <div className="flex-1 overflow-auto">
        <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">

          {/* Title row */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Projects</h1>
              <p className="text-sm text-gray-500 mt-0.5">
                Organise related images and run batch detection
              </p>
            </div>
            <button
              onClick={() => { setShowForm((v) => !v); setFormError('') }}
              className="flex items-center gap-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg font-medium transition-colors"
            >
              <Plus size={14} />
              New Project
            </button>
          </div>

          {/* New project form */}
          {showForm && (
            <div className="bg-white border rounded-xl p-5 space-y-3">
              <p className="text-sm font-medium text-gray-800">Create a new project</p>
              <form onSubmit={handleCreate} className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">Name</label>
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="e.g. Plate 2024-A"
                    className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
                    required
                    autoFocus
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-500 mb-1">
                    Description <span className="text-gray-300">(optional)</span>
                  </label>
                  <textarea
                    value={newDesc}
                    onChange={(e) => setNewDesc(e.target.value)}
                    placeholder="Short description…"
                    rows={2}
                    className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
                  />
                </div>
                {formError && <p className="text-xs text-red-600">{formError}</p>}
                <div className="flex gap-2">
                  <button
                    type="submit"
                    disabled={creating}
                    className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg font-medium disabled:opacity-50 transition-colors"
                  >
                    {creating ? 'Creating…' : 'Create'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowForm(false)}
                    className="px-4 py-1.5 text-sm text-gray-500 hover:text-gray-700"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}

          {/* Projects grid */}
          {loading ? (
            <div className="text-sm text-gray-400 py-12 text-center">Loading…</div>
          ) : projects.length === 0 ? (
            <div className="py-20 text-center space-y-2">
              <FolderOpen size={40} className="mx-auto text-gray-300" />
              <p className="text-sm text-gray-500">No projects yet.</p>
              <p className="text-xs text-gray-400">Create your first project to get started.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {projects.map((p) => (
                <button
                  key={p.id}
                  onClick={() => navigate(`/projects/${p.id}`)}
                  className="bg-white border rounded-xl overflow-hidden hover:border-blue-300 hover:shadow-sm transition-all text-left group"
                >
                  {/* Thumbnail */}
                  <div className="h-36 bg-gray-100 relative overflow-hidden">
                    {p.thumbnail_image_id ? (
                      <img
                        src={thumbnailUrl(p.thumbnail_image_id)}
                        alt=""
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <FolderOpen size={32} className="text-gray-300" />
                      </div>
                    )}
                  </div>

                  {/* Info */}
                  <div className="p-4 space-y-1">
                    <p className="font-medium text-gray-900 truncate">{p.name}</p>
                    {p.description && (
                      <p className="text-xs text-gray-500 truncate">{p.description}</p>
                    )}
                    <p className="text-xs text-gray-400">
                      {p.image_count} image{p.image_count !== 1 ? 's' : ''}{' · '}
                      {p.created_by}{' · '}{fmtDate(p.created_at)}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
