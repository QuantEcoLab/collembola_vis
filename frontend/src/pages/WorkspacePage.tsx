import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogOut, ChevronLeft, ChevronRight, Table2, MousePointer2, Square, XCircle, Pencil, CheckCircle2 } from 'lucide-react'
import ImageUploader from '../components/ImageUploader'
import ImageViewer from '../components/ImageViewer'
import BboxOverlay from '../components/BboxOverlay'
import { AdvancedDetectionModal } from '../components/AdvancedDetectionModal'
import { ViewerToolbar, type OverlayMode } from '../components/ViewerToolbar'
import { MeasurementModal } from '../components/MeasurementModal'
import { WorkspaceSidebar } from '../components/WorkspaceSidebar'
import {
  runDetection,
  runMeasurement,
  imageUrl,
  outputFileUrl,
  updateProjectImageJobs,
  getProject,
  getImage,
} from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import { useRefinement } from '../hooks/useRefinement'
import { useCalibrationStore } from '../store/calibrationStore'
import { useAuthStore } from '../store/authStore'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useProjectStore } from '../store/projectStore'
import type { ImageInfo, ProjectImage } from '../api/types'

type ModalType = 'advanced-detection' | null

export default function WorkspacePage() {
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const role = useAuthStore((s) => s.role)
  const calStore = useCalibrationStore()
  const workspaceStore = useWorkspaceStore()
  const { currentProjectId, currentProjectName } = useProjectStore()

  // ── Core State ──────────────────────────────────────────────────────
  const [image, setImageState] = useState<ImageInfo | null>(workspaceStore.image)
  const setImage = (img: ImageInfo | null) => {
    setImageState(img)
    workspaceStore.setImage(img)
  }

  // ── Workflow State ──────────────────────────────────────────────────
  const [modalOpen, setModalOpen] = useState<ModalType>(null)

  // ── Calibration ─────────────────────────────────────────────────────
  const [umPerPixel, setUmPerPixelState] = useState<number>(calStore.umPerPixel ?? 8.57)
  const setUmPerPixel = (v: number) => {
    setUmPerPixelState(v)
    calStore.setUmManual(v)
  }
  // ── Detection ───────────────────────────────────────────────────────
  const [conf] = useState(0.6)
  const [tileSize] = useState(1280)
  const [overlap] = useState(256)
  const [device] = useState('0')
  const [detectionJobId, setDetectionJobIdState] = useState<string | null>(
    workspaceStore.detectionJobId
  )
  const setDetectionJobId = (id: string | null) => {
    setDetectionJobIdState(id)
    workspaceStore.setDetectionJobId(id)
  }
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const detectionJob = useJobProgress(detectionJobId)
  const detectionDone = detectionJob?.status === 'completed'

  // ── Refinement (Annotation Editing) ─────────────────────────────────
  type EditTool = 'select' | 'reject' | 'draw'
  const [refineMode, setRefineMode] = useState(false)
  const [editTool, setEditTool] = useState<EditTool>('select')
  const [refineSaveError, setRefineSaveError] = useState<string | null>(null)
  const refinement = useRefinement(
    image?.image_id ?? null,
    detectionJobId,
    detectionDone,
  )

  // ── Measurement ─────────────────────────────────────────────────────
  const [measureMethod, setMeasureMethod] = useState<'fast' | 'sam'>('fast')
  const [measureJobId, setMeasureJobIdState] = useState<string | null>(
    workspaceStore.measureJobId
  )
  const setMeasureJobId = (id: string | null) => {
    setMeasureJobIdState(id)
    workspaceStore.setMeasureJobId(id)
  }
  const [measureError, setMeasureError] = useState<string | null>(null)
  const measureJob = useJobProgress(measureJobId)
  const [csvData, setCsvData] = useState<Record<string, any>[] | null>(null)
  const measurementDone = measureJob?.status === 'completed'

  // ── View State ──────────────────────────────────────────────────────
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('raw')
  const prevOverlayModeRef = useRef<OverlayMode>('boxes')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [showMeasurementModal, setShowMeasurementModal] = useState(false)

  // ── Project image list (for next/prev navigation) ──────────────────
  const [projectImages, setProjectImages] = useState<ProjectImage[]>([])

  useEffect(() => {
    if (!currentProjectId) { setProjectImages([]); return }
    getProject(currentProjectId)
      .then((p) => setProjectImages(p.images))
      .catch(() => {})
  }, [currentProjectId])

  const siblingImages = useMemo(() => {
    if (!image || projectImages.length === 0) return []
    const currentEntry = projectImages.find((pi) => pi.image_id === image.image_id)
    const currentFolder = currentEntry?.folder ?? null
    return projectImages.filter((pi) => (pi.folder ?? null) === currentFolder)
  }, [image, projectImages])

  const currentSiblingIndex = useMemo(() => {
    if (!image) return -1
    return siblingImages.findIndex((pi) => pi.image_id === image.image_id)
  }, [image, siblingImages])

  const hasPrev = currentSiblingIndex > 0
  const hasNext = currentSiblingIndex >= 0 && currentSiblingIndex < siblingImages.length - 1

  // ── Derived State ───────────────────────────────────────────────────
  const samOverlayUrl = useMemo(() => {
    if (!measurementDone || !measureJob?.result?.overlay_path) return null
    const filename = (measureJob.result.overlay_path as string).split('/').pop()
    if (!filename) return null
    return outputFileUrl(measureJob.id, filename)
  }, [measurementDone, measureJob])
  const hasSamOverlay = samOverlayUrl !== null

  // Determine which image to show based on overlay mode
  const viewerSrc = useMemo(() => {
    if (!image) return ''
    if ((overlayMode === 'contours' || overlayMode === 'both') && samOverlayUrl) {
      return samOverlayUrl
    }
    return imageUrl(image.image_id, image.filename)
  }, [image, overlayMode, samOverlayUrl])

  // Show bbox overlay when mode is 'boxes' or 'both' and there are boxes to show
  const showBboxOverlay = (overlayMode === 'boxes' || overlayMode === 'both') && refinement.boxes.length > 0

  // Index of selected box within non-added boxes → maps to measurement CSV row
  const selectedMeasurementIndex = useMemo(() => {
    if (!refinement.selectedId) return null
    const box = refinement.boxes.find((b) => b.id === refinement.selectedId)
    if (!box || box.status === 'added') return null
    const origBoxes = refinement.boxes.filter((b) => b.status !== 'added')
    const idx = origBoxes.findIndex((b) => b.id === refinement.selectedId)
    return idx >= 0 ? idx : null
  }, [refinement.selectedId, refinement.boxes])

  // ── Effects ─────────────────────────────────────────────────────────

  // Auto-enter refine mode when saved annotations are restored
  useEffect(() => {
    if (refinement.annotationsSaved && !refineMode) {
      setRefineMode(true)
    }
  }, [refinement.annotationsSaved, refineMode])

  // Auto-show boxes overlay when boxes become available (detection result or saved annotations)
  const boxesAutoShownRef = useRef(false)
  useEffect(() => {
    if (refinement.boxes.length > 0 && !boxesAutoShownRef.current) {
      boxesAutoShownRef.current = true
      setOverlayMode('boxes')
    }
    if (refinement.boxes.length === 0) {
      boxesAutoShownRef.current = false
    }
  }, [refinement.boxes.length])

  // Auto-switch overlay mode when SAM measurement completes
  useEffect(() => {
    if (hasSamOverlay && overlayMode === 'raw') {
      setOverlayMode('contours')
    }
  }, [hasSamOverlay, overlayMode])

  // Load CSV when measurement completes
  useEffect(() => {
    if (measureJob?.status === 'completed' && measureJob.result?.csv_path) {
      const filename = (measureJob.result.csv_path as string).split('/').pop()
      if (!filename) return
      fetch(outputFileUrl(measureJob.id, filename))
        .then((r) => r.text())
        .then((text) => {
          const lines = text.trim().split('\n')
          if (lines.length < 2) return
          const headers = lines[0].split(',').map((h) => h.trim())
          setCsvData(
            lines.slice(1).map((line) => {
              const vals = line.split(',')
              const row: Record<string, unknown> = {}
              headers.forEach((h, i) => {
                const raw = vals[i]?.trim() ?? ''
                const n = Number(raw)
                row[h] = raw !== '' && !isNaN(n) ? n : raw
              })
              return row
            })
          )
        })
        .catch((e) => console.error('[WorkspacePage] Failed to load measurement CSV', e))
    }
  }, [measureJob?.status, measureJob?.result?.csv_path, measureJob?.id])

  // Auto-sync job IDs to project
  useEffect(() => {
    if (!currentProjectId || !image || !detectionJobId) return
    if (detectionJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, {
      detection_job_id: detectionJobId,
    }).catch((e) => console.error('[WorkspacePage] Failed to sync detection job to project', e))
  }, [detectionJob?.status, detectionJobId, currentProjectId, image])

  useEffect(() => {
    if (!currentProjectId || !image || !measureJobId) return
    if (measureJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, {
      measurement_job_id: measureJobId,
    }).catch((e) => console.error('[WorkspacePage] Failed to sync measurement job to project', e))
  }, [measureJob?.status, measureJobId, currentProjectId, image])

  const navigateToImage = useCallback(async (projectImg: ProjectImage) => {
    try {
      const imgInfo = await getImage(projectImg.image_id)
      // Reset all state then load new image (batched by React 18)
      workspaceStore.reset()
      workspaceStore.setImage(imgInfo)
      if (projectImg.detection_job_id) workspaceStore.setDetectionJobId(projectImg.detection_job_id)
      if (projectImg.measurement_job_id) workspaceStore.setMeasureJobId(projectImg.measurement_job_id)
      setImageState(imgInfo)
      setDetectionJobIdState(projectImg.detection_job_id)
      setMeasureJobIdState(projectImg.measurement_job_id)
      setCsvData(null)
      setDetectionError(null)
      setMeasureError(null)
      setRefineMode(false)
      setEditTool('select')
      setRefineSaveError(null)
      setOverlayMode('raw')
      boxesAutoShownRef.current = false
    } catch (e) {
      console.error('[WorkspacePage] Failed to navigate to image', e)
    }
  }, [workspaceStore])

  const handlePrev = useCallback(() => {
    if (!hasPrev) return
    navigateToImage(siblingImages[currentSiblingIndex - 1])
  }, [hasPrev, siblingImages, currentSiblingIndex, navigateToImage])

  const handleNext = useCallback(() => {
    if (!hasNext) return
    navigateToImage(siblingImages[currentSiblingIndex + 1])
  }, [hasNext, siblingImages, currentSiblingIndex, navigateToImage])

  // Keyboard shortcuts: Esc, S, D, H, Delete/Backspace
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      if (e.key === 'Escape') {
        if (editTool !== 'select') {
          setEditTool('select')
          return
        }
        if (refinement.selectedId) {
          refinement.selectBox(null)
          return
        }
        return
      }

      if (!refineMode) return

      if (e.key === 's' || e.key === 'S') {
        setEditTool('select')
        refinement.selectBox(null)
      }
      if (e.key === 'r' || e.key === 'R') {
        setEditTool('reject')
        refinement.selectBox(null)
      }
      if (e.key === 'd' || e.key === 'D') {
        setEditTool('draw')
      }
      if (e.key === 'h' || e.key === 'H') {
        setOverlayMode((prev) => {
          if (prev !== 'raw') { prevOverlayModeRef.current = prev; return 'raw' }
          return prevOverlayModeRef.current
        })
      }

      if (editTool === 'select' && refinement.selectedId) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
          e.preventDefault()
          const sel = refinement.boxes.find((b) => b.id === refinement.selectedId)
          if (sel?.status === 'added') refinement.removeBox(refinement.selectedId)
          else refinement.toggleBox(refinement.selectedId)
        }
      }

      if (!refineMode || editTool !== 'draw') {
        if (e.key === 'ArrowLeft') { e.preventDefault(); handlePrev() }
        if (e.key === 'ArrowRight') { e.preventDefault(); handleNext() }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [editTool, refineMode, refinement, handlePrev, handleNext])

  // ── Handlers ────────────────────────────────────────────────────────

  const handleRunDetection = async (config?: {
    tileSize: number
    overlap: number
    confidence: number
    iouThreshold: number
    device: string
  }) => {
    if (!image) return
    setDetectionError(null)

    const detectionConfig = config
      ? {
          conf: config.confidence,
          iou: config.iouThreshold,
          tile_size: config.tileSize,
          overlap: config.overlap,
          device: config.device === 'auto' ? '0' : config.device,
        }
      : {
          conf,
          iou: 0.5,
          tile_size: tileSize,
          overlap,
          device,
        }

    try {
      const res = await runDetection({
        image_id: image.image_id,
        ...detectionConfig,
      })
      setDetectionJobId(res.job_id)
    } catch (e: any) {
      setDetectionError(e.message)
    }
  }

  const handleDoneEditing = async () => {
    const effectiveJobId = detectionJobId ?? refinement.restoredSourceJobId
    if (!image || !effectiveJobId) return
    setRefineSaveError(null)
    try {
      await refinement.saveAnnotations(image.image_id, image.filename, effectiveJobId)
      setRefineMode(false)
      setEditTool('select')
    } catch (e: any) {
      setRefineSaveError(e.message)
    }
  }

  const handleRunMeasurement = async () => {
    if (!image || !detectionJob?.result?.csv_path) return
    setMeasureError(null)
    setCsvData(null)
    try {
      const res = await runMeasurement({
        image_id: image.image_id,
        detections_csv: detectionJob.result.csv_path,
        um_per_pixel: umPerPixel,
        method: measureMethod,
        device,
        use_annotations: refinement.annotationsSaved,
      })
      setMeasureJobId(res.job_id)
    } catch (e: any) {
      setMeasureError(e.message)
    }
  }

  const handleReset = () => {
    workspaceStore.reset()
    setImageState(null)
    setDetectionJobIdState(null)
    setMeasureJobIdState(null)
    setCsvData(null)
    setDetectionError(null)
    setMeasureError(null)
    setRefineMode(false)
    setEditTool('select')
    setRefineSaveError(null)
    setOverlayMode('raw')
  }

  const handleRowClick = useCallback(
    (rowIndex: number) => {
      const origBoxes = refinement.boxes.filter((b) => b.status !== 'added')
      const box = origBoxes[rowIndex]
      if (!box) return
      refinement.selectBox(box.id)
    },
    [refinement]
  )

  const handleExport = (format: 'image' | 'csv' | 'excel') => {
    if (!measureJob || !measureJob.result) return

    if (format === 'image') {
      // Download the current view
      const link = document.createElement('a')
      link.href = viewerSrc
      link.download = image?.filename ?? 'image.jpg'
      link.click()
    } else if (format === 'csv' && measureJob.result.csv_path) {
      const filename = measureJob.result.csv_path.split('/').pop()!
      const url = outputFileUrl(measureJob.id, filename)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
    } else if (format === 'excel' && measureJob.result.excel_path) {
      const filename = measureJob.result.excel_path.split('/').pop()!
      const url = outputFileUrl(measureJob.id, filename)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
    }
  }


  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      {/* Top bar */}
      <header className="shrink-0 bg-white border-b px-5 h-12 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <span className="font-semibold text-gray-900">Collembola</span>
          <nav className="flex items-center gap-1">
            {currentProjectId && (
              <Link
                to={`/projects/${currentProjectId}`}
                className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
              >
                ← {currentProjectName ?? 'Project'}
              </Link>
            )}
            <Link
              to="/projects"
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Projects
            </Link>
            <Link
              to="/workspace"
              className="text-sm px-3 py-1 rounded-md bg-gray-100 text-gray-900 font-medium"
            >
              Workspace
            </Link>
            {role === 'admin' && (
              <Link
                to="/finetune"
                className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
              >
                Fine-Tune
              </Link>
            )}
          </nav>
          {image && (
            <span
              className="text-xs text-gray-400 truncate max-w-[200px] hidden sm:block"
              title={image.filename}
            >
              {image.filename}
            </span>
          )}
        </div>
        <button
          onClick={() => {
            logout()
            navigate('/login', { replace: true })
          }}
          className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700"
        >
          <LogOut size={14} />
          Sign out
        </button>
      </header>

      {!image ? (
        /* ── Empty state ── */
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-lg space-y-4">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-gray-900">
                Load an image to get started
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                Upload a microscope image or reference one already on the server
              </p>
            </div>
            <ImageUploader onUploaded={setImage} />
          </div>
        </div>
      ) : (
        /* ── Workspace ── */
        <div className="flex flex-1 overflow-hidden">
          {/* ── Sidebar ── */}
          {sidebarCollapsed ? (
            /* Collapsed strip */
            <aside className="w-10 shrink-0 bg-white border-r flex flex-col items-center pt-3 gap-4">
              <button
                onClick={() => setSidebarCollapsed(false)}
                className="text-gray-400 hover:text-gray-700 p-1 rounded hover:bg-gray-100"
                title="Expand sidebar"
              >
                <ChevronRight size={16} />
              </button>
              {/* Mini status dots */}
              <div className="flex flex-col gap-1.5 mt-1">
                {[
                  { done: calStore.umPerPixel != null, label: 'Scale' },
                  { done: detectionDone, label: 'Detect' },
                  { done: refinement.annotationsSaved, label: 'Edit' },
                  { done: measurementDone, label: 'Measure' },
                ].map((s) => (
                  <div
                    key={s.label}
                    className={`w-2.5 h-2.5 rounded-full ${s.done ? 'bg-green-500' : 'bg-gray-200'}`}
                    title={s.label}
                  />
                ))}
              </div>
            </aside>
          ) : (
            /* Expanded sidebar */
            <aside className="w-64 md:w-64 sm:w-full shrink-0 bg-white border-r flex flex-col overflow-hidden max-w-full">
              {/* Sidebar header */}
              <div className="px-4 py-3 border-b flex items-center justify-between">
                <h2 className="font-semibold text-gray-900 text-sm">Workspace</h2>
                <button
                  onClick={() => setSidebarCollapsed(true)}
                  className="text-gray-400 hover:text-gray-700 p-1 rounded hover:bg-gray-100"
                  title="Collapse sidebar"
                >
                  <ChevronRight size={16} className="rotate-180" />
                </button>
              </div>

              <WorkspaceSidebar
                imageFilename={image.filename}
                imageWidth={image.width}
                imageHeight={image.height}
                umPerPixel={umPerPixel}
                setUmPerPixel={setUmPerPixel}
                calibrated={calStore.umPerPixel != null}
                detectionDone={detectionDone}
                detectionJob={detectionJob ?? null}
                detectionError={detectionError}
                onRunDetection={() => handleRunDetection()}
                onAdvancedSettings={() => setModalOpen('advanced-detection')}
                annotationsSaved={refinement.annotationsSaved}
                boxCount={refinement.boxes.length}
                measurementDone={measurementDone}
                measureJob={measureJob ?? null}
                measureError={measureError}
                measureMethod={measureMethod}
                setMeasureMethod={setMeasureMethod}
                onRunMeasurement={handleRunMeasurement}
                csvDataLength={csvData?.length ?? 0}
                onViewResults={() => setShowMeasurementModal(true)}
                onReset={handleReset}
              />
            </aside>
          )}

          {/* ── Main Content ── */}
          <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
            {/* Toolbar row */}
            <div className="flex items-center border-b">
              <div className="flex-1">
                <ViewerToolbar
                  overlayMode={overlayMode}
                  onOverlayChange={setOverlayMode}
                  availableOverlays={{
                    boxes: detectionDone,
                    contours: hasSamOverlay,
                  }}
                  onExport={handleExport}
                  measurementDone={measurementDone}
                />
              </div>
              {siblingImages.length > 1 && (
                <div className="flex items-center gap-0.5 px-2 border-l shrink-0">
                  <button
                    onClick={handlePrev}
                    disabled={!hasPrev}
                    title="Previous image (←)"
                    className="p-1 rounded text-gray-500 hover:text-gray-800 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-default transition-colors"
                  >
                    <ChevronLeft size={15} />
                  </button>
                  <span className="text-xs text-gray-400 tabular-nums px-0.5">
                    {currentSiblingIndex + 1}/{siblingImages.length}
                  </span>
                  <button
                    onClick={handleNext}
                    disabled={!hasNext}
                    title="Next image (→)"
                    className="p-1 rounded text-gray-500 hover:text-gray-800 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-default transition-colors"
                  >
                    <ChevronRight size={15} />
                  </button>
                </div>
              )}
              {measurementDone && (
                <button
                  onClick={() => setShowMeasurementModal(true)}
                  className="flex items-center gap-1.5 text-sm px-3 py-1.5 mr-3 border border-gray-300 rounded-lg bg-white hover:bg-gray-50 text-gray-700 font-medium transition-colors shrink-0"
                >
                  <Table2 size={14} />
                  View Results
                </button>
              )}
            </div>

            {/* Image viewer fills remaining space */}
            <div className="flex-1 min-h-0 relative overflow-hidden">
              <ImageViewer
                src={viewerSrc}
                alt={image.filename}
                className="h-full"
                transformOverlay={showBboxOverlay ? (
                  <BboxOverlay
                    boxes={refinement.boxes}
                    imageWidth={image.width}
                    imageHeight={image.height}
                    selectedId={refinement.selectedId}
                    onBoxClick={refineMode && editTool === 'reject'
                      ? (id) => {
                          const box = refinement.boxes.find((b) => b.id === id)
                          if (box?.status === 'added') refinement.removeBox(id)
                          else refinement.toggleBox(id)
                        }
                      : refinement.selectBox}
                    drawingBox={refinement.drawingBox}
                    mode={refineMode && editTool === 'draw' ? 'draw' : 'review'}
                    onDrawStart={refineMode ? refinement.startDraw : undefined}
                    onDrawMove={refineMode ? refinement.updateDraw : undefined}
                    onDrawEnd={refineMode ? refinement.commitDraw : undefined}
                  />
                ) : undefined}
                disablePan={refineMode && editTool === 'draw'}
              />

              {/* Floating annotation toolbar — always visible once detection is done */}
              {detectionDone && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 flex flex-col gap-0.5 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg p-1.5">
                  {refineMode ? (
                    <>
                      {/* Edit tools */}
                      {([
                        { tool: 'select', label: 'Select', key: 'S', Icon: MousePointer2 },
                        { tool: 'reject', label: 'Invalid', key: 'R', Icon: XCircle },
                        { tool: 'draw',   label: 'Draw',   key: 'D', Icon: Square },
                      ] as const).map(({ tool, label, key, Icon }) => (
                        <button
                          key={tool}
                          title={`${label} (${key})`}
                          onClick={() => {
                            setEditTool(tool)
                            refinement.selectBox(null)
                          }}
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between gap-3 ${
                            editTool === tool
                              ? tool === 'reject'
                                ? 'bg-red-500 text-white'
                                : 'bg-amber-500 text-white'
                              : 'text-gray-600 hover:bg-gray-100'
                          }`}
                        >
                          <span className="flex items-center gap-1.5">
                            <Icon size={12} />
                            {label}
                          </span>
                          <kbd className={`text-[9px] px-1 py-0.5 rounded border font-mono ${
                            editTool === tool
                              ? tool === 'reject'
                                ? 'border-red-300 bg-red-400/40'
                                : 'border-amber-300 bg-amber-400/40'
                              : 'border-gray-200 bg-gray-50 text-gray-400'
                          }`}>
                            {key}
                          </kbd>
                        </button>
                      ))}

                      <div className="border-t border-gray-200 my-0.5" />

                      {/* Hide/Show toggle */}
                      <button
                        onClick={() => setOverlayMode((prev) => {
                          if (prev !== 'raw') { prevOverlayModeRef.current = prev; return 'raw' }
                          return prevOverlayModeRef.current
                        })}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between gap-3 ${
                          overlayMode !== 'raw'
                            ? 'text-gray-600 hover:bg-gray-100'
                            : 'bg-amber-50 text-amber-700 hover:bg-amber-100'
                        }`}
                      >
                        <span>{overlayMode !== 'raw' ? 'Hide' : 'Show'}</span>
                        <kbd className={`text-[9px] px-1 py-0.5 rounded border font-mono ${
                          overlayMode === 'raw' ? 'border-amber-300 bg-amber-400/40' : 'border-gray-200 bg-gray-50 text-gray-400'
                        }`}>
                          H
                        </kbd>
                      </button>

                      <div className="border-t border-gray-200 my-0.5" />

                      {/* Counts */}
                      <div className="flex items-center justify-between gap-2 px-2 py-0.5 text-[10px]">
                        <span className="text-green-600">{refinement.acceptedCount} ✓</span>
                        <span className="text-red-500">{refinement.rejectedCount} ✗</span>
                        <span className="text-blue-600">{refinement.addedCount} +</span>
                      </div>

                      <div className="border-t border-gray-200 my-0.5" />

                      {/* Save & Exit */}
                      <button
                        onClick={handleDoneEditing}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-green-600 text-white hover:bg-green-700 transition-colors flex items-center gap-1.5"
                      >
                        <CheckCircle2 size={12} />
                        Save
                      </button>

                      {/* Fine-tune shortcut (admin only) */}
                      {role === 'admin' && (
                        <button
                          onClick={() => navigate('/finetune')}
                          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-violet-600 text-white hover:bg-violet-700 transition-colors"
                        >
                          Fine-Tune
                        </button>
                      )}

                      {/* Save error */}
                      {refineSaveError && (
                        <p className="text-[10px] text-red-600 px-2 max-w-[140px] leading-tight">{refineSaveError}</p>
                      )}
                    </>
                  ) : (
                    <>
                      {/* Entry state */}
                      {refinement.annotationsSaved && (
                        <div className="flex items-center gap-1 text-[10px] text-green-700 px-2 py-1">
                          <CheckCircle2 size={10} />
                          Annotations saved
                        </div>
                      )}
                      <button
                        onClick={() => setRefineMode(true)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium text-gray-700 hover:bg-gray-100 transition-colors flex items-center gap-1.5"
                      >
                        <Pencil size={12} />
                        {refinement.annotationsSaved ? 'Edit Annotations' : 'Edit Detections'}
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Modals ── */}
      <AdvancedDetectionModal
        isOpen={modalOpen === 'advanced-detection'}
        onClose={() => setModalOpen(null)}
        onSubmit={(config) => {
          handleRunDetection(config)
          setModalOpen(null)
        }}
        initialConfig={{ tileSize, overlap, confidence: conf, iouThreshold: 0.5, device }}
      />

      <MeasurementModal
        isOpen={showMeasurementModal}
        onClose={() => setShowMeasurementModal(false)}
        data={csvData ?? []}
        selectedIndex={selectedMeasurementIndex}
        onRowClick={handleRowClick}
        onExport={(fmt) => handleExport(fmt)}
        measurementDone={measurementDone}
      />
    </div>
  )
}
