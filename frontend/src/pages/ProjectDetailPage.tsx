import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
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
  FolderOpen,
  FolderPlus,
  Pencil,
  List,
  Grid3x3,
  Grid2x2,
  ChevronDown,
} from 'lucide-react'
import {
  getProject,
  addImageToProject,
  removeImageFromProject,
  runBatchProcess,
  thumbnailUrl,
  getImage,
  updateProject,
  deleteProject,
  projectAnnotationsExportUrl,
  listProjectFolders,
  setImageFolder,
  deleteProjectFolder,
  renameProjectFolder,
  downloadProjectResults,
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

function PipelineStatus({ img, compact }: { img: ProjectImage; compact?: boolean }) {
  const steps: PipelineStep[] = [
    { label: 'up',   done: true,                        color: 'bg-gray-400' },
    { label: 'det',  done: !!img.detection_job_id,      color: 'bg-green-500' },
    { label: 'ann',  done: !!img.has_annotation,        color: 'bg-amber-500' },
    { label: 'meas', done: !!img.measurement_job_id,    color: 'bg-blue-500' },
  ]
  return (
    <div className={`flex items-center gap-0.5 ${compact ? 'mt-1' : 'mt-1.5'}`}>
      {steps.map((step, idx) => (
        <div key={step.label} className="flex items-center gap-0.5">
          {idx > 0 && (
            <div className={`flex-1 h-px ${compact ? 'w-2' : 'w-3'} ${steps[idx].done ? step.color : 'bg-gray-200'}`} />
          )}
          <div
            title={step.label}
            className={`${compact ? 'w-1.5 h-1.5' : 'w-2 h-2'} rounded-full shrink-0 ${step.done ? step.color : 'bg-gray-200'}`}
          />
        </div>
      ))}
      {!compact && (
        <div className="flex gap-1 ml-1">
          {steps.map((step) => (
            <span key={step.label} className="text-[9px] text-gray-400 w-5 text-center leading-none">
              {step.label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

type Filter = 'all' | 'needs-det' | 'annotated' | 'done'
type ViewMode = 'carousel' | 'grid' | 'compact' | 'large' | 'list'

export default function ProjectDetailPage() {
  const { id: projectId } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const storedUsername = useAuthStore((s) => s.username)
  const username = storedUsername ?? ''

  const workspaceStore = useWorkspaceStore()
  const { setCurrentProject } = useProjectStore()
  const calibrationUmPerPixel = useCalibrationStore((s) => s.umPerPixel)

  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [showUpload, setShowUpload] = useState(false)
  const [processJobId, setProcessJobId] = useState<string | null>(null)
  const [processError, setProcessError] = useState<string | null>(null)
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
  const [viewMode, setViewMode] = useState<ViewMode>('large')
  const [showViewModeDropdown, setShowViewModeDropdown] = useState(false)
  const viewModeDropdownRef = useRef<HTMLDivElement>(null)

  // Subfolders
  const [folders, setFolders] = useState<string[]>([])
  const [activeFolderTab, setActiveFolderTab] = useState<string | null>(null)
  // Ref so async callbacks always read the current folder without stale closure issues
  const activeFolderTabRef = useRef<string | null>(null)
  activeFolderTabRef.current = activeFolderTab
  const [showNewFolderInput, setShowNewFolderInput] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [folderDropdownFor, setFolderDropdownFor] = useState<string | null>(null)
  const [showBulkFolderDropdown, setShowBulkFolderDropdown] = useState(false)
  const [bulkAssigning, setBulkAssigning] = useState(false)
  const [renamingFolder, setRenamingFolder] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [confirmDeleteFolder, setConfirmDeleteFolder] = useState<string | null>(null)

  const [downloading, setDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)

  const [showProcessOptions, setShowProcessOptions] = useState(false)
  const [processUmPerPixel, setProcessUmPerPixel] = useState<string>('')

  const processJob = useJobProgress(processJobId)

  const load = useCallback(async () => {
    if (!projectId) return
    try {
      const [p, folderList] = await Promise.all([
        getProject(projectId),
        listProjectFolders(projectId).catch(() => [] as string[]),
      ])
      setProject(p)
      setFolders(folderList)
    } catch {
      // auth redirect handled by client
    } finally {
      setLoading(false)
    }
  }, [projectId])

  useEffect(() => { load() }, [load])

  // Close view mode dropdown on outside click
  useEffect(() => {
    if (!showViewModeDropdown) return
    const handler = (e: MouseEvent) => {
      if (viewModeDropdownRef.current && !viewModeDropdownRef.current.contains(e.target as Node)) {
        setShowViewModeDropdown(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showViewModeDropdown])

  // ── Persist process job ID so progress survives page refresh ────────────────
  useEffect(() => {
    if (!projectId) return
    const pid = localStorage.getItem(`collembola_process_${projectId}`)
    if (pid) setProcessJobId(pid)
  }, [projectId])

  useEffect(() => {
    if (!projectId) return
    if (processJobId) localStorage.setItem(`collembola_process_${projectId}`, processJobId)
    else localStorage.removeItem(`collembola_process_${projectId}`)
  }, [projectId, processJobId])

  useEffect(() => {
    if (!projectId || !processJobId) return
    if (processJob?.status === 'completed' || processJob?.status === 'failed') {
      localStorage.removeItem(`collembola_process_${projectId}`)
      load()
    }
  }, [processJob?.status, projectId, processJobId, load])

  // Prefill um/px from calibration store when options panel opens
  useEffect(() => {
    if (showProcessOptions && !processUmPerPixel && calibrationUmPerPixel) {
      setProcessUmPerPixel(String(calibrationUmPerPixel))
    }
  }, [showProcessOptions, calibrationUmPerPixel, processUmPerPixel])

  const isOwner = project?.created_by === username

  // Feature 2 — filtered images (filter + folder)
  const filteredImages = useMemo(() =>
    (project?.images ?? []).filter((img) => {
      // Folder filter
      if (activeFolderTab !== null) {
        if (img.folder !== activeFolderTab) return false
      } else if (folders.length > 0) {
        // Root view with folders: show only ungrouped images in main grid
        if (img.folder != null) return false
      }
      // Status filter
      if (filter === 'needs-det') return !img.detection_job_id
      if (filter === 'annotated') return !!img.has_annotation
      if (filter === 'done') return !!img.has_annotation && !!img.measurement_job_id
      return true
    }),
    [project?.images, activeFolderTab, folders.length, filter]
  )

  const folderCounts = useMemo(() =>
    folders.reduce((acc, f) => {
      acc[f] = (project?.images ?? []).filter((img) => img.folder === f).length
      return acc
    }, {} as Record<string, number>),
    [folders, project?.images]
  )

  const handleUploadDone = async (info: ImageInfo) => {
    if (!projectId) return
    // Read ref — guaranteed to be the current value even if the closure is stale
    const folder = activeFolderTabRef.current
    try {
      await addImageToProject(projectId, info.image_id, info.filename)
    } catch {
      return
    }
    // Assign to active folder before reloading so the folder survives listProjectFolders
    if (folder) {
      try {
        await setImageFolder(projectId, info.image_id, folder)
      } catch {
        // non-fatal: image still added, just not in folder
      }
    }
    await load()
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

  const handleProcessAll = async (umOverride?: number) => {
    if (!projectId) return
    const umVal = umOverride ?? parseFloat(processUmPerPixel)
    if (!umVal || umVal <= 0) return

    setProcessError(null)
    setShowProcessOptions(false)

    try {
      const res = await runBatchProcess(projectId, { um_per_pixel: umVal })
      setProcessJobId(res.job_id)
    } catch (e: any) {
      setProcessError(e.message)
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

  const handleBulkAssignFolder = async (folder: string | null) => {
    if (!projectId || selectedIds.size === 0) return
    setShowBulkFolderDropdown(false)
    setBulkAssigning(true)
    if (folder && !folders.includes(folder)) {
      setFolders((prev) => [...prev, folder].sort())
    }
    // Optimistic update
    setProject((p) => {
      if (!p) return p
      return {
        ...p,
        images: p.images.map((img) =>
          selectedIds.has(img.image_id) ? { ...img, folder } : img
        ),
      }
    })
    try {
      await Promise.all(
        [...selectedIds].map((id) => setImageFolder(projectId, id, folder))
      )
    } catch {
      await load()
    } finally {
      setBulkAssigning(false)
    }
  }

  const handleCreateFolder = () => {
    const name = newFolderName.trim()
    if (!name || folders.includes(name)) return
    setFolders((prev) => [...prev, name].sort())
    setNewFolderName('')
    setShowNewFolderInput(false)
  }

  const handleRenameFolder = async (oldName: string) => {
    const newName = renameValue.trim()
    setRenamingFolder(null)
    setRenameValue('')
    if (!newName || newName === oldName || !projectId) return
    // Optimistic update
    setFolders((prev) => prev.map((f) => f === oldName ? newName : f).sort())
    setProject((p) => {
      if (!p) return p
      return { ...p, images: p.images.map((img) => img.folder === oldName ? { ...img, folder: newName } : img) }
    })
    if (activeFolderTab === oldName) setActiveFolderTab(newName)
    try {
      await renameProjectFolder(projectId, oldName, newName)
    } catch {
      await load()
    }
  }

  const handleDeleteFolder = async (folderName: string) => {
    setConfirmDeleteFolder(null)
    if (!projectId) return
    // Optimistic update: remove folder, unassign images
    setFolders((prev) => prev.filter((f) => f !== folderName))
    setProject((p) => {
      if (!p) return p
      return { ...p, images: p.images.map((img) => img.folder === folderName ? { ...img, folder: null } : img) }
    })
    if (activeFolderTab === folderName) setActiveFolderTab(null)
    try {
      await deleteProjectFolder(projectId, folderName)
    } catch {
      await load()
    }
  }

  const handleAssignFolder = async (imageId: string, folder: string | null) => {
    if (!projectId) return
    setFolderDropdownFor(null)
    // Optimistic update
    setProject((p) => {
      if (!p) return p
      return {
        ...p,
        images: p.images.map((img) =>
          img.image_id === imageId ? { ...img, folder } : img
        ),
      }
    })
    // If assigning a new folder name, add it to folders list
    if (folder && !folders.includes(folder)) {
      setFolders((prev) => [...prev, folder].sort())
    }
    try {
      await setImageFolder(projectId, imageId, folder)
    } catch {
      // Revert on error
      await load()
    }
  }

  const fmtDate = (iso: string) => {
    try { return new Date(iso).toLocaleDateString() } catch { return iso }
  }

  // Feature 1 — summary stats (always from all images, not filtered)
  const allImages = project?.images ?? []
  const measuredCount = allImages.filter((i) => i.measurement_job_id).length

  // Images in current folder/view (no status filter — used for Select All scope)
  const currentFolderImages = allImages.filter((img) => {
    if (activeFolderTab !== null) return img.folder === activeFolderTab
    return folders.length === 0 || img.folder == null
  })

  // Feature 2 — filter counts (folder-aware)
  const filterCounts: Record<Filter, number> = {
    'all': currentFolderImages.length,
    'needs-det': currentFolderImages.filter((i) => !i.detection_job_id).length,
    'annotated': currentFolderImages.filter((i) => i.has_annotation).length,
    'done': currentFolderImages.filter((i) => i.has_annotation && !!i.measurement_job_id).length,
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

      <div className="flex-1 flex overflow-hidden">

        {/* ── Sidebar ──────────────────────────────────────────────────────── */}
        <aside className="w-72 shrink-0 bg-white border-r overflow-y-auto">
          <div className="p-5 space-y-6">

            {/* Back link */}
            <Link
              to="/projects"
              className="inline-flex items-center gap-1 text-sm text-gray-400 hover:text-gray-700"
            >
              <ChevronLeft size={14} />
              Projects
            </Link>

            {/* Project identity */}
            {editing ? (
              <div className="space-y-2">
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="text-base font-semibold border rounded-lg px-3 py-1.5 w-full focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
                <textarea
                  value={editDesc}
                  onChange={(e) => setEditDesc(e.target.value)}
                  rows={2}
                  className="text-sm text-gray-500 border rounded-lg px-3 py-1.5 w-full focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
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
              <div>
                <h1 className="text-base font-semibold text-gray-900">{project.name}</h1>
                {project.description && (
                  <p className="text-sm text-gray-500 mt-0.5">{project.description}</p>
                )}
                <p className="text-xs text-gray-400 mt-1">
                  {project.created_by} · {fmtDate(project.created_at)}
                </p>
              </div>
            )}

            {/* Overview stats */}
            <div>
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-2">Overview</p>
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Images</span>
                  <span className="font-medium text-gray-700">{allImages.length}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-green-600">Detected</span>
                  <span className="font-medium text-green-700">{allImages.filter((i) => i.detection_job_id).length}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-amber-600">Annotated</span>
                  <span className="font-medium text-amber-700">{allImages.filter((i) => i.has_annotation).length}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-blue-600">Measured</span>
                  <span className="font-medium text-blue-700">{measuredCount}</span>
                </div>
                {allImages.reduce((s, i) => s + (i.annotation_accepted ?? 0), 0) > 0 && (
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-teal-600">Accepted boxes</span>
                    <span className="font-medium text-teal-700">
                      {allImages.reduce((s, i) => s + (i.annotation_accepted ?? 0), 0).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Annotations */}
            {(() => {
              const annotated = project.images.filter((i) => i.has_annotation)
              const totalBoxes = annotated.reduce((s, i) => s + i.annotation_total, 0)
              const acceptedBoxesAll = annotated.reduce((s, i) => s + i.annotation_accepted, 0)
              if (annotated.length === 0) return null
              return (
                <div>
                  <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <FileText size={11} className="text-amber-400" />
                    Annotations
                  </p>
                  <div className="space-y-1 mb-2">
                    {project.images.map((img) => (
                      <div key={img.id} className="flex items-center gap-2 text-xs">
                        <div
                          className="w-1.5 h-1.5 rounded-full shrink-0"
                          style={{ background: img.has_annotation ? '#f59e0b' : '#e5e7eb' }}
                        />
                        <span className="flex-1 text-gray-600 truncate" title={img.filename}>
                          {img.filename}
                        </span>
                        {img.has_annotation ? (
                          <span className="text-gray-500 shrink-0">{img.annotation_accepted}</span>
                        ) : (
                          <span className="text-gray-300 shrink-0">–</span>
                        )}
                      </div>
                    ))}
                  </div>
                  <p className="text-[10px] text-gray-400 mb-2">
                    {annotated.length}/{project.images.length} · {acceptedBoxesAll.toLocaleString()} accepted
                    {acceptedBoxesAll !== totalBoxes && (
                      <span className="text-gray-300"> / {totalBoxes.toLocaleString()}</span>
                    )}
                  </p>
                  <div className="flex gap-3">
                    <a
                      href={projectAnnotationsExportUrl(projectId!, 'csv')}
                      download
                      className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium"
                    >
                      <Download size={11} /> CSV
                    </a>
                    <a
                      href={projectAnnotationsExportUrl(projectId!, 'json')}
                      download
                      className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium"
                    >
                      <Download size={11} /> JSON
                    </a>
                  </div>
                </div>
              )
            })()}

            {/* Manage */}
            {isOwner && !editing && (
              <div>
                <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-2">Manage</p>
                <div className="space-y-2">
                  <button
                    onClick={() => {
                      setEditName(project.name)
                      setEditDesc(project.description)
                      setEditing(true)
                    }}
                    className="w-full flex items-center gap-2 text-sm px-3 py-1.5 border rounded-lg text-gray-600 hover:bg-gray-50 transition-colors"
                  >
                    <Pencil size={13} /> Edit project
                  </button>
                  {confirmDelete ? (
                    <div className="space-y-2">
                      <p className="text-xs text-red-600 font-medium">Delete this project?</p>
                      <div className="flex gap-2">
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
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={() => setConfirmDelete(true)}
                      className="w-full flex items-center gap-2 text-sm px-3 py-1.5 border border-red-200 rounded-lg text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <Trash2 size={13} /> Delete project
                    </button>
                  )}
                </div>
              </div>
            )}

          </div>
        </aside>

        {/* ── Main area ────────────────────────────────────────────────────── */}
        <main className="flex-1 overflow-auto bg-gray-50">
          <div className="px-6 py-6 space-y-5">

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

                  {currentFolderImages.length > 0 && (
                    <button
                      onClick={() => {
                        const allInFolder = currentFolderImages.map((i) => i.image_id)
                        const allSelected = allInFolder.every((id) => selectedIds.has(id))
                        if (allSelected) {
                          setSelectedIds(new Set())
                        } else {
                          setSelectedIds(new Set(allInFolder))
                        }
                      }}
                      className="flex items-center gap-1.5 text-sm bg-white border rounded-lg px-3 py-1.5 text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      {currentFolderImages.every((i) => selectedIds.has(i.image_id))
                        ? <><Square size={14} /> Deselect All</>
                        : <><CheckSquare size={14} /> Select All</>
                      }
                    </button>
                  )}

                  {selectedIds.size > 0 && (
                    <>
                      {/* Move to folder */}
                      <div className="relative">
                        <button
                          onClick={() => setShowBulkFolderDropdown((v) => !v)}
                          disabled={bulkAssigning}
                          className="flex items-center gap-1.5 text-sm bg-white border border-purple-300 rounded-lg px-3 py-1.5 text-purple-700 hover:bg-purple-50 font-medium disabled:opacity-50 transition-colors"
                        >
                          {bulkAssigning
                            ? <><Loader2 size={13} className="animate-spin" /> Moving…</>
                            : <><FolderOpen size={13} /> Move to folder</>
                          }
                        </button>
                        {showBulkFolderDropdown && (
                          <>
                            <div className="fixed inset-0 z-10" onClick={() => setShowBulkFolderDropdown(false)} />
                            <div className="absolute left-0 top-full mt-1 w-52 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-20">
                              {/* Inline new folder option */}
                              {showNewFolderInput ? (
                                <div className="px-3 py-2 flex items-center gap-1 border-b">
                                  <input
                                    autoFocus
                                    type="text"
                                    value={newFolderName}
                                    onChange={(e) => setNewFolderName(e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter' && newFolderName.trim()) {
                                        const name = newFolderName.trim()
                                        setNewFolderName('')
                                        setShowNewFolderInput(false)
                                        handleBulkAssignFolder(name)
                                      }
                                      if (e.key === 'Escape') { setShowNewFolderInput(false); setNewFolderName('') }
                                    }}
                                    placeholder="New folder name"
                                    className="flex-1 text-xs border border-purple-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-purple-400"
                                  />
                                  <button
                                    onClick={() => {
                                      const name = newFolderName.trim()
                                      if (!name) return
                                      setNewFolderName('')
                                      setShowNewFolderInput(false)
                                      handleBulkAssignFolder(name)
                                    }}
                                    disabled={!newFolderName.trim()}
                                    className="text-xs text-purple-600 font-medium disabled:opacity-40"
                                  >
                                    Add
                                  </button>
                                </div>
                              ) : (
                                <button
                                  onClick={() => setShowNewFolderInput(true)}
                                  className="w-full px-3 py-2 text-left text-xs text-purple-600 hover:bg-purple-50 flex items-center gap-1.5 border-b"
                                >
                                  <FolderPlus size={12} /> New folder…
                                </button>
                              )}
                              <button
                                onClick={() => handleBulkAssignFolder(null)}
                                className="w-full px-3 py-2 text-left text-xs text-gray-500 hover:bg-gray-50"
                              >
                                (Remove from folder)
                              </button>
                              {folders.map((f) => (
                                <button
                                  key={f}
                                  onClick={() => handleBulkAssignFolder(f)}
                                  className="w-full px-3 py-2 text-left text-xs text-gray-700 hover:bg-purple-50 flex items-center gap-1.5"
                                >
                                  <FolderOpen size={11} /> {f}
                                </button>
                              ))}
                            </div>
                          </>
                        )}
                      </div>

                      {confirmBulkRemove ? (
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
                      )}
                    </>
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
                        onClick={() => {
                          if (calibrationUmPerPixel) {
                            handleProcessAll(calibrationUmPerPixel)
                          } else {
                            setShowProcessOptions((v) => !v)
                          }
                        }}
                        disabled={processJob?.status === 'running' || processJob?.status === 'pending'}
                        className="flex items-center gap-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-3 py-1.5 font-medium disabled:opacity-50 transition-colors"
                      >
                        <Play size={13} />
                        Process All
                      </button>

                      {measuredCount > 0 && (
                        <button
                          onClick={async () => {
                            setDownloading(true)
                            setDownloadError(null)
                            try {
                              await downloadProjectResults(projectId!, project.name)
                            } catch (e: any) {
                              setDownloadError(e.message)
                            } finally {
                              setDownloading(false)
                            }
                          }}
                          disabled={downloading}
                          className="flex items-center gap-1.5 text-sm bg-white border border-teal-300 rounded-lg px-3 py-1.5 text-teal-700 hover:bg-teal-50 font-medium disabled:opacity-50 transition-colors"
                        >
                          {downloading
                            ? <><Loader2 size={13} className="animate-spin" /> Downloading…</>
                            : <><Download size={13} /> Download Results</>
                          }
                        </button>
                      )}

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
                  onClick={() => handleProcessAll()}
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

            {/* Process All job progress */}
            {processJob && (processJob.status === 'running' || processJob.status === 'pending') && (
              <JobProgress job={processJob} />
            )}
            {processJob?.status === 'completed' && (
              <div className="flex items-center gap-2 text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-2">
                <CheckCircle2 size={15} />
                Processing complete — all images detected and measured
              </div>
            )}
            {processJob?.status === 'failed' && (
              <p className="text-sm text-red-600">{processJob.error}</p>
            )}
            {processError && <p className="text-sm text-red-600">{processError}</p>}
            {downloadError && <p className="text-sm text-red-600">Download failed: {downloadError}</p>}

            {/* Image section */}
            {project.images.length === 0 ? (
              <div className="py-20 text-center text-sm text-gray-400 bg-white border rounded-xl">
                No images yet. Click <strong>Add Images</strong> to upload.
              </div>
            ) : (
              <div>
                {/* Folder cards grid — root view */}
                {activeFolderTab === null && (
                  <div className="mb-6">
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Folders</p>
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
                      {folders.map((f) => (
                        <div
                          key={f}
                          className="relative bg-white border rounded-xl group hover:border-amber-300 hover:bg-amber-50 transition-colors"
                        >
                          {/* Confirm-delete overlay */}
                          {confirmDeleteFolder === f ? (
                            <div className="flex flex-col items-center justify-center gap-2 p-4 h-full min-h-[110px]">
                              <p className="text-xs text-red-600 font-medium text-center">Delete "{f}"?</p>
                              <p className="text-[10px] text-gray-400 text-center">Images become ungrouped</p>
                              <div className="flex gap-2">
                                <button
                                  onClick={() => handleDeleteFolder(f)}
                                  className="text-xs bg-red-600 text-white px-2 py-1 rounded hover:bg-red-700 font-medium"
                                >
                                  Delete
                                </button>
                                <button
                                  onClick={() => setConfirmDeleteFolder(null)}
                                  className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1"
                                >
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : renamingFolder === f ? (
                            /* Rename inline */
                            <div className="flex flex-col items-center justify-center gap-2 p-3 min-h-[110px]">
                              <FolderOpen size={22} className="text-amber-400" />
                              <input
                                autoFocus
                                type="text"
                                value={renameValue}
                                onChange={(e) => setRenameValue(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') handleRenameFolder(f)
                                  if (e.key === 'Escape') { setRenamingFolder(null); setRenameValue('') }
                                }}
                                className="text-xs border border-amber-300 rounded px-2 py-1 w-full focus:outline-none focus:ring-1 focus:ring-amber-400"
                              />
                              <div className="flex gap-2">
                                <button
                                  onClick={() => handleRenameFolder(f)}
                                  disabled={!renameValue.trim()}
                                  className="text-xs text-amber-700 font-medium disabled:opacity-40 hover:text-amber-900"
                                >
                                  Save
                                </button>
                                <button
                                  onClick={() => { setRenamingFolder(null); setRenameValue('') }}
                                  className="text-xs text-gray-400 hover:text-gray-600"
                                >
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            /* Normal card */
                            <>
                              <button
                                onClick={() => { setActiveFolderTab(f); setSelectedIds(new Set()) }}
                                className="flex flex-col items-center justify-center gap-2 p-4 w-full text-center min-h-[110px]"
                              >
                                <FolderOpen size={28} className="text-amber-400 group-hover:text-amber-500" />
                                <span className="text-xs font-medium text-gray-700 truncate w-full">{f}</span>
                                <span className="text-[10px] text-gray-400">
                                  {folderCounts[f] ?? 0} image{(folderCounts[f] ?? 0) !== 1 ? 's' : ''}
                                </span>
                              </button>
                              {/* Hover action buttons */}
                              <div className="absolute top-1.5 right-1.5 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button
                                  onClick={(e) => { e.stopPropagation(); setRenamingFolder(f); setRenameValue(f) }}
                                  className="p-1 rounded text-gray-400 hover:text-amber-600 hover:bg-amber-100 transition-colors"
                                  title="Rename folder"
                                >
                                  <Pencil size={11} />
                                </button>
                                <button
                                  onClick={(e) => { e.stopPropagation(); setConfirmDeleteFolder(f) }}
                                  className="p-1 rounded text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors"
                                  title="Delete folder"
                                >
                                  <Trash2 size={11} />
                                </button>
                              </div>
                            </>
                          )}
                        </div>
                      ))}
                      {/* New Folder dashed card — hidden in select mode */}
                      {!selectMode && (
                        showNewFolderInput ? (
                          <div className="flex flex-col items-center justify-center gap-2 bg-white border-2 border-dashed border-purple-200 rounded-xl p-3">
                            <FolderPlus size={22} className="text-purple-300" />
                            <input
                              autoFocus
                              type="text"
                              value={newFolderName}
                              onChange={(e) => setNewFolderName(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') handleCreateFolder()
                                if (e.key === 'Escape') { setShowNewFolderInput(false); setNewFolderName('') }
                              }}
                              placeholder="Folder name"
                              className="text-xs border border-purple-300 rounded px-2 py-1 w-full focus:outline-none focus:ring-1 focus:ring-purple-400"
                            />
                            <div className="flex gap-2">
                              <button
                                onClick={handleCreateFolder}
                                disabled={!newFolderName.trim()}
                                className="text-xs text-purple-600 font-medium disabled:opacity-40 hover:text-purple-800"
                              >
                                Add
                              </button>
                              <button
                                onClick={() => { setShowNewFolderInput(false); setNewFolderName('') }}
                                className="text-xs text-gray-400 hover:text-gray-600"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <button
                            onClick={() => setShowNewFolderInput(true)}
                            className="flex flex-col items-center justify-center gap-2 bg-white border-2 border-dashed border-gray-200 rounded-xl p-4 hover:border-purple-300 hover:bg-purple-50 transition-colors group"
                          >
                            <FolderPlus size={28} className="text-gray-300 group-hover:text-purple-400" />
                            <span className="text-xs text-gray-400 group-hover:text-purple-600">New Folder</span>
                          </button>
                        )
                      )}
                    </div>
                  </div>
                )}

                {/* Breadcrumb — when inside a folder */}
                {activeFolderTab !== null && (
                  <div className="flex items-center gap-1.5 mb-4 text-sm">
                    <button
                      onClick={() => { setActiveFolderTab(null); setSelectedIds(new Set()) }}
                      className="flex items-center gap-1 text-gray-500 hover:text-gray-800 transition-colors"
                    >
                      <ChevronLeft size={14} />
                      All Folders
                    </button>
                    <span className="text-gray-300">/</span>
                    <span className="flex items-center gap-1.5 text-gray-800 font-medium">
                      <FolderOpen size={14} className="text-amber-400" />
                      {activeFolderTab}
                    </span>
                  </div>
                )}

                {/* Controls row: filter pills (left) + view mode toggle (right) */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-1.5 flex-wrap">
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
                  <div className="relative" ref={viewModeDropdownRef}>
                    <button
                      onClick={() => setShowViewModeDropdown((v) => !v)}
                      className="flex items-center gap-1 text-gray-600 hover:text-gray-900 text-sm px-2 py-1 rounded-lg hover:bg-gray-100 transition-colors"
                      title="Change view"
                    >
                      {viewMode === 'carousel' && <AlignJustify size={15} />}
                      {viewMode === 'grid' && <LayoutGrid size={15} />}
                      {viewMode === 'compact' && <Grid3x3 size={15} />}
                      {viewMode === 'large' && <Grid2x2 size={15} />}
                      {viewMode === 'list' && <List size={15} />}
                      <ChevronDown size={13} className={`transition-transform ${showViewModeDropdown ? 'rotate-180' : ''}`} />
                    </button>
                    {showViewModeDropdown && (
                      <div className="absolute right-0 top-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-1 min-w-[140px]">
                        {[
                          { mode: 'grid' as ViewMode, label: 'Grid', icon: LayoutGrid },
                          { mode: 'compact' as ViewMode, label: 'Compact', icon: Grid3x3 },
                          { mode: 'large' as ViewMode, label: 'Large', icon: Grid2x2 },
                          { mode: 'carousel' as ViewMode, label: 'Carousel', icon: AlignJustify },
                          { mode: 'list' as ViewMode, label: 'List', icon: List },
                        ].map(({ mode, label, icon: Icon }) => (
                          <button
                            key={mode}
                            onClick={() => {
                              setViewMode(mode)
                              setShowViewModeDropdown(false)
                            }}
                            className={`w-full flex items-center gap-2 px-3 py-1.5 text-sm transition-colors ${
                              viewMode === mode
                                ? 'bg-blue-50 text-blue-700 font-medium'
                                : 'text-gray-700 hover:bg-gray-50'
                            }`}
                          >
                            <Icon size={14} />
                            {label}
                            {viewMode === mode && (
                              <CheckCircle2 size={13} className="ml-auto" />
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {filteredImages.length === 0 ? (
                  <p className="text-sm text-gray-400 py-8 text-center">No images match this filter.</p>
                ) : (
                  <div className={
                    viewMode === 'carousel'
                      ? 'flex gap-4 overflow-x-auto pb-3 scroll-smooth'
                      : viewMode === 'compact'
                      ? 'grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2'
                      : viewMode === 'large'
                      ? 'grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-4'
                      : viewMode === 'list'
                      ? 'flex flex-col gap-2'
                      : 'grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3'
                  }>
                    {filteredImages.map((img) => {
                      const isSelected = selectedIds.has(img.image_id)

                      // List view - horizontal layout
                      if (viewMode === 'list') {
                        return (
                          <div
                            key={img.id}
                            className={`bg-white border rounded-lg overflow-hidden flex items-center gap-3 p-2 group relative transition-all ${
                              isSelected ? 'ring-2 ring-blue-500 border-blue-500' : 'hover:border-gray-300'
                            }`}
                          >
                            {/* Thumbnail */}
                            <button
                              onClick={() => selectMode ? toggleSelect(img.image_id) : handleOpenInWorkspace(img)}
                              className="block w-20 h-20 bg-gray-100 relative overflow-hidden rounded flex-shrink-0"
                              title={selectMode ? (isSelected ? 'Deselect' : 'Select') : 'Open in Workspace'}
                            >
                              <img
                                src={thumbnailUrl(img.image_id)}
                                alt={img.filename}
                                className={`w-full h-full object-cover transition-transform duration-300 ${
                                  selectMode ? '' : 'group-hover:scale-105'
                                } ${isSelected ? 'opacity-70' : ''}`}
                              />
                              {selectMode && (
                                <div className={`absolute inset-0 flex items-center justify-center transition-colors ${
                                  isSelected ? 'bg-blue-500/20' : 'hover:bg-gray-900/10'
                                }`}>
                                  {isSelected && (
                                    <div className="bg-blue-600 rounded-full p-0.5">
                                      <CheckCircle2 size={16} className="text-white" />
                                    </div>
                                  )}
                                </div>
                              )}
                              {!img.detection_job_id && processJob?.status === 'running' && (
                                <div className="absolute bottom-1 left-1">
                                  <span className="bg-gray-500 text-white text-[9px] font-medium px-1 py-0.5 rounded flex items-center gap-0.5">
                                    <Loader2 size={8} className="animate-spin" />
                                    …
                                  </span>
                                </div>
                              )}
                            </button>

                            {/* Info */}
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-gray-900 font-medium truncate" title={img.filename}>
                                {img.filename}
                              </p>
                              <p className="text-xs text-gray-400">{img.added_by}</p>
                              <PipelineStatus img={img} />
                            </div>

                            {/* Actions */}
                            {!selectMode && (
                              <div className="flex items-center gap-1 shrink-0">
                                <div className="relative">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      setFolderDropdownFor(
                                        folderDropdownFor === img.image_id ? null : img.image_id
                                      )
                                    }}
                                    className="text-gray-300 hover:text-purple-500 transition-colors p-1"
                                    title="Assign to folder"
                                  >
                                    <FolderOpen size={14} />
                                  </button>
                                  {folderDropdownFor === img.image_id && (
                                    <>
                                      <div
                                        className="fixed inset-0 z-10"
                                        onClick={() => setFolderDropdownFor(null)}
                                      />
                                      <div className="absolute right-0 top-full mt-1 w-40 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-20">
                                        <button
                                          onClick={() => handleAssignFolder(img.image_id, null)}
                                          className={`w-full px-3 py-1.5 text-left text-xs hover:bg-gray-50 ${
                                            img.folder == null ? 'text-gray-400' : 'text-gray-700'
                                          }`}
                                        >
                                          (No folder)
                                        </button>
                                        {folders.map((f) => (
                                          <button
                                            key={f}
                                            onClick={() => handleAssignFolder(img.image_id, f)}
                                            className={`w-full px-3 py-1.5 text-left text-xs hover:bg-gray-50 ${
                                              img.folder === f ? 'text-purple-600 font-medium' : 'text-gray-700'
                                            }`}
                                          >
                                            {f}
                                          </button>
                                        ))}
                                      </div>
                                    </>
                                  )}
                                </div>
                                <button
                                  onClick={() => handleRemoveImage(img)}
                                  disabled={removingId === img.image_id}
                                  className="text-gray-300 hover:text-red-500 disabled:opacity-40 transition-colors p-1"
                                  title="Remove from project"
                                >
                                  {removingId === img.image_id ? (
                                    <Loader2 size={14} className="animate-spin" />
                                  ) : (
                                    <Trash2 size={14} />
                                  )}
                                </button>
                              </div>
                            )}
                          </div>
                        )
                      }

                      // Grid/Carousel view - vertical card layout
                      return (
                        <div
                          key={img.id}
                          className={`${
                            viewMode === 'carousel' ? 'shrink-0 w-44'
                            : viewMode === 'compact' ? 'w-full'
                            : viewMode === 'large' ? 'w-full'
                            : 'w-full'
                          } bg-white border rounded-xl overflow-visible group relative transition-all ${
                            isSelected ? 'ring-2 ring-blue-500 border-blue-500' : ''
                          }`}
                        >
                          {/* Thumbnail */}
                          <button
                            onClick={() => selectMode ? toggleSelect(img.image_id) : handleOpenInWorkspace(img)}
                            className={`block w-full ${
                              viewMode === 'compact' ? 'h-24'
                              : viewMode === 'large' ? 'h-48'
                              : 'h-32'
                            } bg-gray-100 relative overflow-hidden rounded-t-xl`}
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
                            {!img.detection_job_id && processJob?.status === 'running' && (
                              <div className="absolute bottom-1.5 left-1.5">
                                <span className="bg-gray-500 text-white text-[10px] font-medium px-1.5 py-0.5 rounded flex items-center gap-0.5">
                                  <Loader2 size={8} className="animate-spin" />
                                  …
                                </span>
                              </div>
                            )}
                          </button>

                          {/* Filename + contributor + remove */}
                          <div className={viewMode === 'compact' ? 'px-1.5 py-1.5' : viewMode === 'large' ? 'px-3 py-3' : 'px-2 py-2'}>
                            <div className="flex items-center justify-between gap-1">
                              <div className="flex-1 min-w-0">
                                <p className={`${viewMode === 'compact' ? 'text-[10px]' : viewMode === 'large' ? 'text-sm' : 'text-xs'} text-gray-600 truncate`} title={img.filename}>
                                  {img.filename}
                                </p>
                                {viewMode !== 'compact' && (
                                  <p className="text-[10px] text-gray-400 truncate">{img.added_by}</p>
                                )}
                              </div>
                              {!selectMode && (
                                <div className="flex items-center gap-1 shrink-0">
                                  {/* Folder assignment button */}
                                  <div className="relative">
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        setFolderDropdownFor(
                                          folderDropdownFor === img.image_id ? null : img.image_id
                                        )
                                      }}
                                      className="text-gray-300 hover:text-purple-500 transition-colors"
                                      title="Assign to folder"
                                    >
                                      <FolderOpen size={12} />
                                    </button>
                                    {folderDropdownFor === img.image_id && (
                                      <>
                                        <div
                                          className="fixed inset-0 z-10"
                                          onClick={() => setFolderDropdownFor(null)}
                                        />
                                        <div className="absolute right-0 bottom-full mb-1 w-40 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-20">
                                          <button
                                            onClick={() => handleAssignFolder(img.image_id, null)}
                                            className={`w-full px-3 py-1.5 text-left text-xs hover:bg-gray-50 ${
                                              img.folder == null ? 'text-gray-400' : 'text-gray-700'
                                            }`}
                                          >
                                            (No folder)
                                          </button>
                                          {folders.map((f) => (
                                            <button
                                              key={f}
                                              onClick={() => handleAssignFolder(img.image_id, f)}
                                              className={`w-full px-3 py-1.5 text-left text-xs hover:bg-purple-50 flex items-center gap-1.5 ${
                                                img.folder === f ? 'text-purple-700 font-medium' : 'text-gray-700'
                                              }`}
                                            >
                                              <FolderOpen size={10} />
                                              {f}
                                            </button>
                                          ))}
                                        </div>
                                      </>
                                    )}
                                  </div>
                                  <button
                                    onClick={() => handleRemoveImage(img)}
                                    disabled={removingId === img.image_id}
                                    className="text-gray-300 hover:text-red-500 transition-colors disabled:opacity-50"
                                    title="Remove from project"
                                  >
                                    {removingId === img.image_id
                                      ? <Loader2 size={12} className="animate-spin" />
                                      : <Trash2 size={12} />
                                    }
                                  </button>
                                </div>
                              )}
                            </div>
                            {/* Feature 4 — Pipeline status */}
                            <PipelineStatus img={img} compact={viewMode === 'compact'} />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}

          </div>
        </main>
      </div>
    </div>
  )
}
