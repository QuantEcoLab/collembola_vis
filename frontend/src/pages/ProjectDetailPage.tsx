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
  CheckSquare,
  Square,
  LayoutGrid,
  AlignJustify,
  Ruler,
} from 'lucide-react'
import {
  getProject,
  addImageToProject,
  removeImageFromProject,
  runBatchDetection,
  runBatchMeasurement,
  thumbnailUrl,
  getImage,
  updateProject,
  deleteProject,
  projectAnnotationsExportUrl,
} from '../api/client'
import { useAuthStore } from '../store/authStore'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useProjectStore } from '../store/projectStore'
import { useCalibrationStore } from '../store/calibrationStore'
import { useJobProgress } from '../hooks/useJob'
import ImageUploader from '../components/ImageUploader'
import JobProgress from '../components/JobProgress'
import type { ProjectDetail, ProjectImage, ImageInfo } from '../api/types'

// ── Pipeline status indicator ────────────────────────────────────────────────

type PipelineStep = { label: string; done: boolean; color: string }

function PipelineStatus({ img }: { img: ProjectImage }) {
  const steps: PipelineStep[] = [
    { label: 'up',   done: true,                        color: 'bg-gray-400' },
    { label: 'det',  done: !!img.detection_job_id,      color: 'bg-green-500' },
    { label: 'ann',  done: !!img.has_annotation,        color: 'bg-amber-500' },
    { label: 'meas', done: !!img.measurement_job_id,    color: 'bg-blue-500' },
  ]
  return (
    <div className="flex items-center gap-0.5 mt-1.5">
      {steps.map((step, idx) => (
        <div key={step.label} className="flex items-center gap-0.5">
          {idx > 0 && (
            <div className={`flex-1 h-px w-3 ${steps[idx].done ? step.color : 'bg-gray-200'}`} />
          )}
          <div
            title={step.label}
            className={`w-2 h-2 rounded-full shrink-0 ${step.done ? step.color : 'bg-gray-200'}`}
          />
        </div>
      ))}
      <div className="flex gap-1 ml-1">
        {steps.map((step) => (
          <span key={step.label} className="text-[9px] text-gray-400 w-5 text-center leading-none">
            {step.label}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

type Filter = 'all' | 'needs-det' | 'annotated' | 'done'
type ViewMode = 'carousel' | 'grid'

export default function ProjectDetailPage() {
  const { id: projectId } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const storedUsername = useAuthStore((s) => s.username)
  const token = useAuthStore((s) => s.token)
  const username = storedUsername ?? (() => {
    try {
      if (!token) return ''
      const payload = JSON.parse(atob(token.split('.')[1]))
      return (payload.sub as string) ?? ''
    } catch { return '' }
  })()

  const workspaceStore = useWorkspaceStore()
  const { setCurrentProject } = useProjectStore()
  const calibrationUmPerPixel = useCalibrationStore((s) => s.umPerPixel)

  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [showUpload, setShowUpload] = useState(false)
  const [batchJobId, setBatchJobId] = useState<string | null>(null)
  const [batchError, setBatchError] = useState<string | null>(null)
  const [removingId, setRemovingId] = useState<string | null>(null)
  const [selectMode, setSelectMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [bulkRemoving, setBulkRemoving] = useState(false)
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [confirmBulkRemove, setConfirmBulkRemove] = useState(false)

  // Feature 2 — filter
  const [filter, setFilter] = useState<Filter>('all')

  // Feature 3 — view mode
  const [viewMode, setViewMode] = useState<ViewMode>('carousel')

  // Feature 5 — batch measure
  const [measureBatchJobId, setMeasureBatchJobId] = useState<string | null>(null)
  const [measureBatchError, setMeasureBatchError] = useState<string | null>(null)
  const [showProcessOptions, setShowProcessOptions] = useState(false)
  const [processUmPerPixel, setProcessUmPerPixel] = useState<string>('')
  const [autoMeasureAfterDetect, setAutoMeasureAfterDetect] = useState(false)

  const batchJob = useJobProgress(batchJobId)
  const measureBatchJob = useJobProgress(measureBatchJobId)

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

  // Prefill um/px from calibration store when options open
  useEffect(() => {
    if (showProcessOptions && !processUmPerPixel && calibrationUmPerPixel) {
      setProcessUmPerPixel(String(calibrationUmPerPixel))
    }
  }, [showProcessOptions, calibrationUmPerPixel, processUmPerPixel])

  // Refresh project after batch detect completes + auto-trigger measurement
  useEffect(() => {
    if (batchJob?.status !== 'completed') return
    load()
    
    // Auto-trigger measurement if this was a "Process All" operation
    if (autoMeasureAfterDetect && projectId) {
      setAutoMeasureAfterDetect(false)
      const umVal = parseFloat(processUmPerPixel)
      if (umVal && umVal > 0) {
        setMeasureBatchError(null)
        runBatchMeasurement(projectId, { um_per_pixel: umVal })
          .then((res) => setMeasureBatchJobId(res.job_id))
          .catch((e) => setMeasureBatchError(e.message))
      }
    }
  }, [batchJob?.status, load, autoMeasureAfterDetect, projectId, processUmPerPixel])

  // Refresh project after batch measure completes
  useEffect(() => {
    if (measureBatchJob?.status === 'completed') {
      load()
    }
  }, [measureBatchJob?.status, load])

  const isOwner = project?.created_by === username

  // Feature 2 — filtered images (filter doesn't affect stats)
  const filteredImages = (project?.images ?? []).filter((img) => {
    if (filter === 'needs-det') return !img.detection_job_id
    if (filter === 'annotated') return !!img.has_annotation
    if (filter === 'done') return !!img.has_annotation && !!img.measurement_job_id
    return true
  })

  const handleUploadDone = async (info: ImageInfo) => {
    if (!projectId) return
    try {
      await addImageToProject(projectId, info.image_id, info.filename)
      await load()
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

  const toggleSelectMode = () => {
    setSelectMode((v) => !v)
    setSelectedIds(new Set())
    setConfirmBulkRemove(false)
  }

  const toggleSelect = (imageId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(imageId)) next.delete(imageId)
      else next.add(imageId)
      return next
    })
  }

  const handleBulkRemove = async () => {
    if (!projectId || selectedIds.size === 0) return
    setBulkRemoving(true)
    setConfirmBulkRemove(false)
    try {
      await Promise.all([...selectedIds].map((id) => removeImageFromProject(projectId, id)))
      setSelectedIds(new Set())
      setSelectMode(false)
      await load()
    } catch {
      // ignore
    } finally {
      setBulkRemoving(false)
    }
  }

  const handleProcessAll = async () => {
    if (!projectId) return
    const umVal = parseFloat(processUmPerPixel)
    if (!umVal || umVal <= 0) return
    
    setBatchError(null)
    setMeasureBatchError(null)
    setShowProcessOptions(false)
    setAutoMeasureAfterDetect(true)
    
    try {
      const res = await runBatchDetection(projectId, {})
      setBatchJobId(res.job_id)
    } catch (e: any) {
      setBatchError(e.message)
      setAutoMeasureAfterDetect(false)
    }
  }

  const handleOpenInWorkspace = async (img: ProjectImage) => {
    try {
      const info = await getImage(img.image_id)
      setCurrentProject(projectId!, project!.name)
      workspaceStore.setImage(info)
      workspaceStore.setDetectionJobId(img.detection_job_id ?? null)
      workspaceStore.setMeasureJobId(img.measurement_job_id ?? null)
      navigate('/workspace')
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
    if (!projectId) return
    setDeleting(true)
    setConfirmDelete(false)
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

  // Feature 1 — summary stats (always from all images, not filtered)
  const allImages = project?.images ?? []
  const detectedCount = allImages.filter((i) => i.detection_job_id).length
  const annotatedCount = allImages.filter((i) => i.has_annotation).length
  const measuredCount = allImages.filter((i) => i.measurement_job_id).length
  const acceptedBoxes = allImages.reduce((s, i) => s + (i.annotation_accepted ?? 0), 0)

  // Feature 2 — filter counts
  const filterCounts: Record<Filter, number> = {
    'all': allImages.length,
    'needs-det': allImages.filter((i) => !i.detection_job_id).length,
    'annotated': allImages.filter((i) => i.has_annotation).length,
    'done': allImages.filter((i) => i.has_annotation && !!i.measurement_job_id).length,
  }

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
                    Created by {project.created_by} · {fmtDate(project.created_at)}
                  </p>
                </div>
                {isOwner && (
                  <div className="flex gap-2 shrink-0 items-center">
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
                    {confirmDelete ? (
                      <>
                        <span className="text-xs text-red-600 font-medium">Delete project?</span>
                        <button
                          onClick={handleDelete}
                          disabled={deleting}
                          className="text-xs px-3 py-1.5 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
                        >
                          {deleting ? 'Deleting…' : 'Yes, delete'}
                        </button>
                        <button
                          onClick={() => setConfirmDelete(false)}
                          className="text-xs px-3 py-1.5 text-gray-500 hover:text-gray-700"
                        >
                          Cancel
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={() => setConfirmDelete(true)}
                        className="text-xs px-3 py-1.5 border border-red-200 rounded-lg text-red-600 hover:bg-red-50"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Action bar */}
          <div className="flex items-center gap-3 flex-wrap">
            {selectMode ? (
              <>
                <button
                  onClick={toggleSelectMode}
                  className="flex items-center gap-1.5 text-sm bg-white border rounded-lg px-3 py-1.5 text-gray-500 hover:bg-gray-50 transition-colors"
                >
                  <X size={14} />
                  Cancel
                </button>

                {project.images.length > 0 && (
                  <button
                    onClick={() => {
                      if (selectedIds.size === project.images.length) {
                        setSelectedIds(new Set())
                      } else {
                        setSelectedIds(new Set(project.images.map((i) => i.image_id)))
                      }
                    }}
                    className="flex items-center gap-1.5 text-sm bg-white border rounded-lg px-3 py-1.5 text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    {selectedIds.size === project.images.length
                      ? <><Square size={14} /> Deselect All</>
                      : <><CheckSquare size={14} /> Select All</>
                    }
                  </button>
                )}

                {selectedIds.size > 0 && (
                  confirmBulkRemove ? (
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-red-600 font-medium">
                        Remove {selectedIds.size} image{selectedIds.size > 1 ? 's' : ''}?
                      </span>
                      <button
                        onClick={handleBulkRemove}
                        disabled={bulkRemoving}
                        className="flex items-center gap-1.5 text-sm bg-red-600 hover:bg-red-700 text-white rounded-lg px-3 py-1.5 font-medium disabled:opacity-50 transition-colors"
                      >
                        {bulkRemoving
                          ? <><Loader2 size={13} className="animate-spin" /> Removing…</>
                          : <><Trash2 size={13} /> Confirm</>
                        }
                      </button>
                      <button
                        onClick={() => setConfirmBulkRemove(false)}
                        className="text-sm text-gray-500 hover:text-gray-700 px-2"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setConfirmBulkRemove(true)}
                      className="flex items-center gap-1.5 text-sm bg-red-600 hover:bg-red-700 text-white rounded-lg px-3 py-1.5 font-medium transition-colors"
                    >
                      <Trash2 size={13} /> Remove {selectedIds.size} image{selectedIds.size > 1 ? 's' : ''}
                    </button>
                  )
                )}
              </>
            ) : (
              <>
                <button
                  onClick={() => setShowUpload((v) => !v)}
                  className="flex items-center gap-1.5 text-sm bg-white border rounded-lg px-3 py-1.5 text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  <Plus size={14} />
                  Add Images
                </button>

                {project.images.length > 0 && (
                  <>
                    <button
                      onClick={() => setShowProcessOptions((v) => !v)}
                      disabled={batchJob?.status === 'running' || batchJob?.status === 'pending' || measureBatchJob?.status === 'running' || measureBatchJob?.status === 'pending'}
                      className="flex items-center gap-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-3 py-1.5 font-medium disabled:opacity-50 transition-colors"
                    >
                      <Play size={13} />
                      Process All
                    </button>

                    <button
                      onClick={toggleSelectMode}
                      className="flex items-center gap-1.5 text-sm bg-white border rounded-lg px-3 py-1.5 text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <CheckSquare size={14} />
                      Select
                    </button>
                  </>
                )}
              </>
            )}
          </div>

          {/* Process options inline row */}
          {showProcessOptions && (
            <div className="flex items-center gap-3 bg-blue-50 border border-blue-200 rounded-lg px-4 py-3">
              <Ruler size={14} className="text-blue-600 shrink-0" />
              <span className="text-sm text-blue-800 font-medium shrink-0">Calibration (µm/px):</span>
              <input
                type="number"
                step="0.01"
                min="0.01"
                value={processUmPerPixel}
                onChange={(e) => setProcessUmPerPixel(e.target.value)}
                placeholder="e.g. 8.57"
                className="w-28 text-sm border border-blue-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white"
              />
              <button
                onClick={handleProcessAll}
                disabled={!parseFloat(processUmPerPixel)}
                className="text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md px-3 py-1 font-medium disabled:opacity-40 transition-colors"
              >
                Start Processing
              </button>
              <button
                onClick={() => setShowProcessOptions(false)}
                className="text-gray-400 hover:text-gray-600 ml-1"
              >
                <X size={14} />
              </button>
              <span className="text-xs text-blue-600 ml-auto">
                Detect → Measure {project.images.length} image{project.images.length !== 1 ? 's' : ''}
              </span>
            </div>
          )}

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

          {/* Batch detect job progress */}
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

          {/* Batch measure job progress */}
          {measureBatchJob && (measureBatchJob.status === 'running' || measureBatchJob.status === 'pending') && (
            <JobProgress job={measureBatchJob} />
          )}
          {measureBatchJob?.status === 'completed' && (
            <div className="flex items-center gap-2 text-sm text-teal-700 bg-teal-50 border border-teal-200 rounded-lg px-4 py-2">
              <CheckCircle2 size={15} />
              Batch measurement complete —{' '}
              {(measureBatchJob.result as any)?.num_measured ?? 0} images measured
            </div>
          )}
          {measureBatchJob?.status === 'failed' && (
            <p className="text-sm text-red-600">{measureBatchJob.error}</p>
          )}
          {measureBatchError && <p className="text-sm text-red-600">{measureBatchError}</p>}

          {/* Image section */}
          {project.images.length === 0 ? (
            <div className="py-20 text-center text-sm text-gray-400 bg-white border rounded-xl">
              No images yet. Click <strong>Add Images</strong> to upload.
            </div>
          ) : (
            <div>
              {/* Section header: Images label + summary stats + view toggle */}
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Images
                </p>
                {/* Feature 3 — view mode toggle */}
                <button
                  onClick={() => setViewMode((v) => v === 'carousel' ? 'grid' : 'carousel')}
                  className="text-gray-400 hover:text-gray-700 transition-colors"
                  title={viewMode === 'carousel' ? 'Switch to grid' : 'Switch to carousel'}
                >
                  {viewMode === 'carousel' ? <LayoutGrid size={15} /> : <AlignJustify size={15} />}
                </button>
              </div>

              {/* Feature 1 — Summary stats bar */}
              <div className="flex items-center gap-2 text-xs mb-3 flex-wrap">
                <span className="text-gray-500">{allImages.length} image{allImages.length !== 1 ? 's' : ''}</span>
                <span className="text-gray-300">·</span>
                <span className="text-green-600">{detectedCount} detected</span>
                <span className="text-gray-300">·</span>
                <span className="text-amber-600">{annotatedCount} annotated</span>
                <span className="text-gray-300">·</span>
                <span className="text-blue-600">{measuredCount} measured</span>
                {acceptedBoxes > 0 && (
                  <>
                    <span className="text-gray-300">·</span>
                    <span className="text-teal-600">{acceptedBoxes.toLocaleString()} accepted boxes</span>
                  </>
                )}
              </div>

              {/* Feature 2 — Filter bar */}
              <div className="flex items-center gap-1.5 mb-4 flex-wrap">
                {(['all', 'needs-det', 'annotated', 'done'] as Filter[]).map((f) => {
                  const labels: Record<Filter, string> = {
                    'all': 'All',
                    'needs-det': 'Needs detection',
                    'annotated': 'Annotated',
                    'done': 'Done',
                  }
                  const active = filter === f
                  return (
                    <button
                      key={f}
                      onClick={() => {
                        setFilter(f)
                        setSelectedIds(new Set())
                      }}
                      className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                        active
                          ? 'bg-gray-800 text-white border-gray-800'
                          : 'bg-white text-gray-500 border-gray-200 hover:border-gray-400'
                      }`}
                    >
                      {labels[f]} <span className={active ? 'text-gray-300' : 'text-gray-400'}>{filterCounts[f]}</span>
                    </button>
                  )
                })}
              </div>

              {filteredImages.length === 0 ? (
                <p className="text-sm text-gray-400 py-8 text-center">No images match this filter.</p>
              ) : (
                <div className={
                  viewMode === 'carousel'
                    ? 'flex gap-4 overflow-x-auto pb-3'
                    : 'grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3'
                }>
                  {filteredImages.map((img) => {
                    const isSelected = selectedIds.has(img.image_id)
                    return (
                      <div
                        key={img.id}
                        className={`${viewMode === 'carousel' ? 'shrink-0 w-44' : 'w-full'} bg-white border rounded-xl overflow-hidden group relative transition-all ${
                          isSelected ? 'ring-2 ring-blue-500 border-blue-500' : ''
                        }`}
                      >
                        {/* Thumbnail */}
                        <button
                          onClick={() => selectMode ? toggleSelect(img.image_id) : handleOpenInWorkspace(img)}
                          className="block w-full h-32 bg-gray-100 relative overflow-hidden"
                          title={selectMode ? (isSelected ? 'Deselect' : 'Select') : 'Open in Workspace'}
                        >
                          <img
                            src={thumbnailUrl(img.image_id)}
                            alt={img.filename}
                            className={`w-full h-full object-cover transition-transform duration-300 ${
                              selectMode ? '' : 'group-hover:scale-105'
                            } ${isSelected ? 'opacity-70' : ''}`}
                          />

                          {/* Selection overlay */}
                          {selectMode && (
                            <div className={`absolute inset-0 flex items-center justify-center transition-colors ${
                              isSelected ? 'bg-blue-500/20' : 'hover:bg-gray-900/10'
                            }`}>
                              {isSelected && (
                                <div className="bg-blue-600 rounded-full p-1">
                                  <CheckCircle2 size={20} className="text-white" />
                                </div>
                              )}
                            </div>
                          )}

                          {/* Running indicator */}
                          {!img.detection_job_id && batchJob?.status === 'running' && (
                            <div className="absolute bottom-1.5 left-1.5">
                              <span className="bg-gray-500 text-white text-[10px] font-medium px-1.5 py-0.5 rounded flex items-center gap-0.5">
                                <Loader2 size={8} className="animate-spin" />
                                …
                              </span>
                            </div>
                          )}
                        </button>

                        {/* Filename + contributor + remove */}
                        <div className="px-2 py-2">
                          <div className="flex items-center justify-between gap-1">
                            <div className="flex-1 min-w-0">
                              <p className="text-xs text-gray-600 truncate" title={img.filename}>
                                {img.filename}
                              </p>
                              <p className="text-[10px] text-gray-400 truncate">{img.added_by}</p>
                            </div>
                            {!selectMode && (
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
                            )}
                          </div>
                          {/* Feature 4 — Pipeline status */}
                          <PipelineStatus img={img} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {/* Annotations management — shown once at least one image has annotations */}
          {(() => {
            const annotated = project.images.filter((i) => i.has_annotation)
            const totalBoxes = annotated.reduce((s, i) => s + i.annotation_total, 0)
            const acceptedBoxesAll = annotated.reduce((s, i) => s + i.annotation_accepted, 0)
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
                    <span className="font-medium text-gray-700">{acceptedBoxesAll.toLocaleString()}</span> accepted boxes
                    {acceptedBoxesAll !== totalBoxes && (
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
