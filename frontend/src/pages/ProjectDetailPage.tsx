import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useParams, Link } from 'react-router-dom'
import {
  LogOut,
  ChevronLeft,
  Play,
  Plus,
  Trash2,
  X,
  CheckCircle2,
  Loader2,
  Download,
  FileText,
} from 'lucide-react'
import {
  getProject,
  addImageToProject,
  removeImageFromProject,
  runBatchDetection,
  thumbnailUrl,
  getImage,
  updateProject,
  deleteProject,
  projectAnnotationsExportUrl,
} from '../api/client'
import { useAuthStore } from '../store/authStore'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useProjectStore } from '../store/projectStore'
import { useJobProgress } from '../hooks/useJob'
import ImageUploader from '../components/ImageUploader'
import JobProgress from '../components/JobProgress'
import type { ProjectDetail, ProjectImage, ImageInfo } from '../api/types'

export default function ProjectDetailPage() {
  const { id: projectId } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const storedUsername = useAuthStore((s) => s.username)
  const token = useAuthStore((s) => s.token)
  // Derive username from store or fall back to decoding the JWT (for existing sessions)
  const username = storedUsername ?? (() => {
    try {
      if (!token) return ''
      const payload = JSON.parse(atob(token.split('.')[1]))
      return (payload.sub as string) ?? ''
    } catch { return '' }
  })()

  const workspaceStore = useWorkspaceStore()
  const { setCurrentProject } = useProjectStore()

  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [showUpload, setShowUpload] = useState(false)
  const [batchJobId, setBatchJobId] = useState<string | null>(null)
  const [batchError, setBatchError] = useState<string | null>(null)
  const [removingId, setRemovingId] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)

  const batchJob = useJobProgress(batchJobId)

  const load = useCallback(async () => {
    if (!projectId) return
    try {
      const p = await getProject(projectId)
      setProject(p)
    } catch {
      // auth redirect handled by client
    } finally {
      setLoading(false)
    }
  }, [projectId])

  useEffect(() => { load() }, [load])

  // Refresh project after batch completes to pick up new detection_job_ids
  useEffect(() => {
    if (batchJob?.status === 'completed') {
      load()
    }
  }, [batchJob?.status, load])

  const isOwner = project?.created_by === username

  const handleUploadDone = async (info: ImageInfo) => {
    if (!projectId) return
    try {
      await addImageToProject(projectId, info.image_id, info.filename)
      await load()
      // Keep upload panel open so user can add more files
    } catch {
      // ignore
    }
  }

  const handleRemoveImage = async (img: ProjectImage) => {
    setRemovingId(img.image_id)
    try {
      await removeImageFromProject(projectId!, img.image_id)
      await load()
    } catch {
      // ignore
    } finally {
      setRemovingId(null)
    }
  }

  const handleBatchDetect = async () => {
    if (!projectId) return
    setBatchError(null)
    try {
      const res = await runBatchDetection(projectId, {})
      setBatchJobId(res.job_id)
    } catch (e: any) {
      setBatchError(e.message)
    }
  }

  const handleOpenInWorkspace = async (img: ProjectImage) => {
    try {
      const info = await getImage(img.image_id)
      setCurrentProject(projectId!, project!.name)
      workspaceStore.setImage(info)
      if (img.detection_job_id) {
        workspaceStore.setDetectionJobId(img.detection_job_id)
      } else {
        workspaceStore.setDetectionJobId(null)
      }
      workspaceStore.setMeasureJobId(null)
      navigate('/')
    } catch {
      // ignore
    }
  }

  const handleSaveEdit = async () => {
    if (!projectId) return
    setSaving(true)
    try {
      await updateProject(projectId, editName, editDesc)
      setProject((p) => p ? { ...p, name: editName, description: editDesc } : p)
      setEditing(false)
    } catch {
      // ignore
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!projectId || !window.confirm(`Delete project "${project?.name}"? This cannot be undone.`)) return
    setDeleting(true)
    try {
      await deleteProject(projectId)
      navigate('/projects')
    } catch {
      setDeleting(false)
    }
  }

  const fmtDate = (iso: string) => {
    try { return new Date(iso).toLocaleDateString() } catch { return iso }
  }

  const detectedCount = project?.images.filter((i) => i.detection_job_id).length ?? 0

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-gray-400">
        Loading…
      </div>
    )
  }

  if (!project) {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-gray-400">
        Project not found.{' '}
        <Link to="/projects" className="ml-2 text-blue-500 hover:underline">Back to projects</Link>
      </div>
    )
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
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Projects
            </Link>
            <Link
              to="/"
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
        <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">

          {/* Back + title */}
          <div>
            <Link
              to="/projects"
              className="inline-flex items-center gap-1 text-sm text-gray-400 hover:text-gray-700 mb-3"
            >
              <ChevronLeft size={14} />
              Projects
            </Link>

            {editing ? (
              <div className="space-y-2">
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="text-xl font-semibold border rounded-lg px-3 py-1.5 w-full max-w-md focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
                <textarea
                  value={editDesc}
                  onChange={(e) => setEditDesc(e.target.value)}
                  rows={2}
                  className="text-sm text-gray-500 border rounded-lg px-3 py-1.5 w-full max-w-md focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveEdit}
                    disabled={saving}
                    className="text-sm px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {saving ? 'Saving…' : 'Save'}
                  </button>
                  <button
                    onClick={() => setEditing(false)}
                    className="text-sm px-3 py-1.5 text-gray-500 hover:text-gray-700"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-start gap-3">
                <div className="flex-1">
                  <h1 className="text-xl font-semibold text-gray-900">{project.name}</h1>
                  {project.description && (
                    <p className="text-sm text-gray-500 mt-0.5">{project.description}</p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">
                    Created by {project.created_by} · {fmtDate(project.created_at)} ·{' '}
                    {project.images.length} image{project.images.length !== 1 ? 's' : ''},{' '}
                    {detectedCount} detected
                  </p>
                </div>
                {isOwner && (
                  <div className="flex gap-2 shrink-0">
                    <button
                      onClick={() => {
                        setEditName(project.name)
                        setEditDesc(project.description)
                        setEditing(true)
                      }}
                      className="text-xs px-3 py-1.5 border rounded-lg text-gray-600 hover:bg-gray-50"
                    >
                      Edit
                    </button>
                    <button
                      onClick={handleDelete}
                      disabled={deleting}
                      className="text-xs px-3 py-1.5 border border-red-200 rounded-lg text-red-600 hover:bg-red-50 disabled:opacity-50"
                    >
                      {deleting ? 'Deleting…' : 'Delete'}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Action bar */}
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => setShowUpload((v) => !v)}
              className="flex items-center gap-1.5 text-sm bg-white border rounded-lg px-3 py-1.5 text-gray-700 hover:bg-gray-50 transition-colors"
            >
              <Plus size={14} />
              Add Images
            </button>

            {project.images.length > 0 && (
              <button
                onClick={handleBatchDetect}
                disabled={batchJob?.status === 'running' || batchJob?.status === 'pending'}
                className="flex items-center gap-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-3 py-1.5 font-medium disabled:opacity-50 transition-colors"
              >
                <Play size={13} />
                Run Detection on All
              </button>
            )}
          </div>

          {/* Upload area */}
          {showUpload && (
            <div className="bg-white border rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-medium text-gray-800">Add image to project</p>
                <button onClick={() => setShowUpload(false)} className="text-gray-400 hover:text-gray-600">
                  <X size={16} />
                </button>
              </div>
              <ImageUploader onUploaded={handleUploadDone} multiple />
            </div>
          )}

          {/* Batch job progress */}
          {batchJob && (batchJob.status === 'running' || batchJob.status === 'pending') && (
            <JobProgress job={batchJob} />
          )}
          {batchJob?.status === 'completed' && (
            <div className="flex items-center gap-2 text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-2">
              <CheckCircle2 size={15} />
              Batch detection complete —{' '}
              {(batchJob.result as any)?.results?.length ?? 0} images processed
            </div>
          )}
          {batchJob?.status === 'failed' && (
            <p className="text-sm text-red-600">{batchJob.error}</p>
          )}
          {batchError && <p className="text-sm text-red-600">{batchError}</p>}

          {/* Image carousel */}
          {project.images.length === 0 ? (
            <div className="py-20 text-center text-sm text-gray-400 bg-white border rounded-xl">
              No images yet. Click <strong>Add Images</strong> to upload.
            </div>
          ) : (
            <div>
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Images
              </p>
              <div className="flex gap-4 overflow-x-auto pb-3">
                {project.images.map((img) => (
                  <div
                    key={img.id}
                    className="shrink-0 w-44 bg-white border rounded-xl overflow-hidden group relative"
                  >
                    {/* Thumbnail */}
                    <button
                      onClick={() => handleOpenInWorkspace(img)}
                      className="block w-full h-32 bg-gray-100 relative overflow-hidden"
                      title="Open in Workspace"
                    >
                      <img
                        src={thumbnailUrl(img.image_id)}
                        alt={img.filename}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      {/* Status badges */}
                      <div className="absolute bottom-1.5 left-1.5 flex gap-1 flex-wrap">
                        {img.detection_job_id && (
                          <span className="bg-green-600 text-white text-[10px] font-medium px-1.5 py-0.5 rounded">
                            ✓ det
                          </span>
                        )}
                        {img.has_annotation && (
                          <span className="bg-amber-500 text-white text-[10px] font-medium px-1.5 py-0.5 rounded">
                            ✓ annot
                          </span>
                        )}
                        {img.measurement_job_id && (
                          <span className="bg-blue-600 text-white text-[10px] font-medium px-1.5 py-0.5 rounded">
                            ✓ meas
                          </span>
                        )}
                        {!img.detection_job_id && batchJob?.status === 'running' && (
                          <span className="bg-gray-500 text-white text-[10px] font-medium px-1.5 py-0.5 rounded flex items-center gap-0.5">
                            <Loader2 size={8} className="animate-spin" />
                            …
                          </span>
                        )}
                      </div>
                    </button>

                    {/* Filename + contributor + remove */}
                    <div className="px-2 py-2 flex items-center justify-between gap-1">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-gray-600 truncate" title={img.filename}>
                          {img.filename}
                        </p>
                        <p className="text-[10px] text-gray-400 truncate">{img.added_by}</p>
                      </div>
                      <button
                        onClick={() => handleRemoveImage(img)}
                        disabled={removingId === img.image_id}
                        className="text-gray-300 hover:text-red-500 transition-colors shrink-0 disabled:opacity-50"
                        title="Remove from project"
                      >
                        {removingId === img.image_id
                          ? <Loader2 size={12} className="animate-spin" />
                          : <Trash2 size={12} />
                        }
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Annotations management — shown once at least one image has annotations */}
          {(() => {
            const annotated = project.images.filter((i) => i.has_annotation)
            const totalBoxes = annotated.reduce((s, i) => s + i.annotation_total, 0)
            const acceptedBoxes = annotated.reduce((s, i) => s + i.annotation_accepted, 0)
            if (annotated.length === 0) return null
            return (
              <div className="bg-white border rounded-xl p-5 space-y-4">
                <div className="flex items-center gap-2">
                  <FileText size={16} className="text-amber-500" />
                  <p className="text-sm font-medium text-gray-800">Annotations</p>
                </div>

                {/* Per-image annotation rows */}
                <div className="space-y-1.5">
                  {project.images.map((img) => (
                    <div
                      key={img.id}
                      className="flex items-center gap-3 text-xs"
                    >
                      <div className="w-2 h-2 rounded-full shrink-0" style={{
                        background: img.has_annotation ? '#f59e0b' : '#e5e7eb'
                      }} />
                      <span className="flex-1 text-gray-600 truncate" title={img.filename}>
                        {img.filename}
                      </span>
                      {img.has_annotation ? (
                        <span className="text-gray-500 shrink-0">
                          {img.annotation_accepted} accepted
                          {img.annotation_total !== img.annotation_accepted && (
                            <span className="text-gray-400"> / {img.annotation_total} total</span>
                          )}
                        </span>
                      ) : (
                        <span className="text-gray-300 shrink-0">no annotations</span>
                      )}
                    </div>
                  ))}
                </div>

                {/* Summary + export */}
                <div className="flex items-center justify-between pt-2 border-t border-gray-100">
                  <p className="text-xs text-gray-500">
                    {annotated.length}/{project.images.length} images annotated
                    {' · '}
                    <span className="font-medium text-gray-700">{acceptedBoxes.toLocaleString()}</span> accepted boxes
                    {acceptedBoxes !== totalBoxes && (
                      <span className="text-gray-400"> / {totalBoxes.toLocaleString()} total</span>
                    )}
                  </p>
                  <div className="flex gap-2">
                    <a
                      href={projectAnnotationsExportUrl(projectId!, 'csv')}
                      download
                      className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium"
                    >
                      <Download size={12} />
                      CSV
                    </a>
                    <a
                      href={projectAnnotationsExportUrl(projectId!, 'json')}
                      download
                      className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium"
                    >
                      <Download size={12} />
                      JSON
                    </a>
                  </div>
                </div>
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
