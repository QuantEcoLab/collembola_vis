import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogOut, RefreshCw, ChevronRight, Settings2 } from 'lucide-react'
import ImageUploader from '../components/ImageUploader'
import ImageViewer from '../components/ImageViewer'
import BboxOverlay from '../components/BboxOverlay'
import JobProgress from '../components/JobProgress'
import MeasurementTable from '../components/MeasurementTable'
import { StepCard } from '../components/StepCard'
import { ManualCalibrationModal } from '../components/ManualCalibrationModal'
import { AdvancedDetectionModal } from '../components/AdvancedDetectionModal'
import { FineTuneModal } from '../components/FineTuneModal'
import { ViewerToolbar, type OverlayMode } from '../components/ViewerToolbar'
import { PathDecisionCard } from '../components/PathDecisionCard'
import { NavigationWarningDialog } from '../components/NavigationWarningDialog'
import {
  runDetection,
  runMeasurement,
  imageUrl,
  outputFileUrl,
  updateProjectImageJobs,
} from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import { useRefinement } from '../hooks/useRefinement'
import { useCalibrationStore } from '../store/calibrationStore'
import { useAuthStore } from '../store/authStore'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useProjectStore } from '../store/projectStore'
import type { ImageInfo } from '../api/types'

type WorkflowStep = 1 | 2 | 3 | 4
type ModalType = 'manual-calibration' | 'advanced-detection' | 'finetune' | null

// Common presets for calibration
const CALIBRATION_PRESETS = [
  { label: '8.57 μm/px (default)', value: 8.57 },
  { label: '10.0 μm/px', value: 10.0 },
  { label: '5.0 μm/px', value: 5.0 },
]

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
  const [currentStep, setCurrentStep] = useState<WorkflowStep>(1)
  const [annotationPath, setAnnotationPath] = useState(false) // User chose to edit annotations?
  const [modalOpen, setModalOpen] = useState<ModalType>(null)

  // ── Calibration ─────────────────────────────────────────────────────
  const [umPerPixel, setUmPerPixelState] = useState<number>(calStore.umPerPixel ?? 8.57)
  const setUmPerPixel = (v: number) => {
    setUmPerPixelState(v)
    calStore.setUmManual(v)
  }
  const [calibrateError, setCalibrateError] = useState<string | null>(null)

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
  const [refineMode, setRefineMode] = useState(false)
  const [drawMode, setDrawMode] = useState(false)
  const [refineSaveError, setRefineSaveError] = useState<string | null>(null)
  const refinement = useRefinement(
    image?.image_id ?? null,
    detectionDone ? detectionJobId : null
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
  const [splitPercent, setSplitPercent] = useState(58)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const mainAreaRef = useRef<HTMLDivElement>(null)

  // ── Navigation Warning ──────────────────────────────────────────────
  const [navigationTarget, setNavigationTarget] = useState<WorkflowStep | null>(null)
  const [showNavigationWarning, setShowNavigationWarning] = useState(false)

  // ── Derived State ───────────────────────────────────────────────────
  const hasSamOverlay = measurementDone && !!measureJob?.result?.overlay_path
  const samOverlayUrl = hasSamOverlay
    ? outputFileUrl(measureJob!.id, measureJob!.result.overlay_path.split('/').pop()!)
    : null

  // Determine which image to show based on overlay mode
  const viewerSrc = useMemo(() => {
    if (!image) return ''
    if ((overlayMode === 'contours' || overlayMode === 'both') && samOverlayUrl) {
      return samOverlayUrl
    }
    return imageUrl(image.image_id, image.filename)
  }, [image, overlayMode, samOverlayUrl])

  // Show bbox overlay when mode is 'boxes' or 'both'
  const showBboxOverlay = (overlayMode === 'boxes' || overlayMode === 'both') && detectionDone

  // Workflow steps for dynamic stepper
  const workflowSteps = useMemo(() => {
    if (annotationPath) {
      return [
        { number: 1, label: 'Scale', done: calStore.umPerPixel != null },
        { number: 2, label: 'Detect', done: detectionDone },
        { number: 3, label: 'Edit', done: refinement.annotationsSaved },
        { number: 4, label: 'Measure', done: measurementDone },
      ]
    } else {
      return [
        { number: 1, label: 'Scale', done: calStore.umPerPixel != null },
        { number: 2, label: 'Detect', done: detectionDone },
        { number: 4, label: 'Measure', done: measurementDone },
      ]
    }
  }, [annotationPath, calStore.umPerPixel, detectionDone, refinement.annotationsSaved, measurementDone])

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
      setAnnotationPath(true)
      setCurrentStep(3)
    }
  }, [refinement.annotationsSaved, refineMode])

  // Auto-switch overlay mode when SAM measurement completes
  useEffect(() => {
    if (hasSamOverlay && overlayMode === 'raw') {
      setOverlayMode('contours')
    }
  }, [hasSamOverlay, overlayMode])

  // Load CSV when measurement completes
  useEffect(() => {
    if (measureJob?.status === 'completed' && measureJob.result?.csv_path) {
      const filename = measureJob.result.csv_path.split('/').pop()!
      fetch(outputFileUrl(measureJob.id, filename))
        .then((r) => r.text())
        .then((text) => {
          const lines = text.trim().split('\n')
          const headers = lines[0].split(',')
          setCsvData(
            lines.slice(1).map((line) => {
              const vals = line.split(',')
              const row: Record<string, any> = {}
              headers.forEach((h, i) => {
                const n = Number(vals[i])
                row[h] = isNaN(n) ? vals[i] : n
              })
              return row
            })
          )
        })
        .catch(() => {})
    }
  }, [measureJob?.status, measureJob?.result?.csv_path, measureJob?.id])

  // Auto-sync job IDs to project
  useEffect(() => {
    if (!currentProjectId || !image || !detectionJobId) return
    if (detectionJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, {
      detection_job_id: detectionJobId,
    }).catch(() => {})
  }, [detectionJob?.status, detectionJobId, currentProjectId, image])

  useEffect(() => {
    if (!currentProjectId || !image || !measureJobId) return
    if (measureJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, {
      measurement_job_id: measureJobId,
    }).catch(() => {})
  }, [measureJob?.status, measureJobId, currentProjectId, image])

  // Listen for calibration point events from modal
  useEffect(() => {
    const handleStart = () => {
      // Modal is ready to receive points
    }

    const handleCancel = () => {
      // Modal cancelled point selection
    }

    const handleComplete = () => {
      // Modal finished point selection
    }

    window.addEventListener('start-calibration-point-selection', handleStart)
    window.addEventListener('cancel-calibration-point-selection', handleCancel)
    window.addEventListener('complete-calibration-point-selection', handleComplete)

    return () => {
      window.removeEventListener('start-calibration-point-selection', handleStart)
      window.removeEventListener('cancel-calibration-point-selection', handleCancel)
      window.removeEventListener('complete-calibration-point-selection', handleComplete)
    }
  }, [])

  // Keyboard shortcuts: Esc, S, D, H, Delete/Backspace
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      if (e.key === 'Escape') {
        if (drawMode) {
          setDrawMode(false)
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
        setDrawMode(false)
        refinement.selectBox(null)
      }
      if (e.key === 'd' || e.key === 'D') {
        setDrawMode(true)
      }
      if (e.key === 'h' || e.key === 'H') {
        // Toggle annotations visibility (handled by overlay mode)
      }

      if (!drawMode && refinement.selectedId) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
          e.preventDefault()
          const sel = refinement.boxes.find((b) => b.id === refinement.selectedId)
          if (sel?.status === 'added') refinement.removeBox(refinement.selectedId)
          else refinement.toggleBox(refinement.selectedId)
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [drawMode, refineMode, refinement])

  // ── Handlers ────────────────────────────────────────────────────────

  const handleManualCalibrationComplete = (calculatedUmPerPixel: number) => {
    setUmPerPixel(calculatedUmPerPixel)
    calStore.setCalibration(calculatedUmPerPixel, '', 'manual', 1.0)
    setCurrentStep(2) // Advance to detection step
  }

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

  const handlePathDecision = (path: 'annotate' | 'measure') => {
    setAnnotationPath(path === 'annotate')
    if (path === 'annotate') {
      setRefineMode(true)
      setCurrentStep(3)
    } else {
      setCurrentStep(4)
    }
  }

  const handleDoneEditing = async () => {
    const effectiveJobId = detectionJobId ?? refinement.restoredSourceJobId
    if (!image || !effectiveJobId) return
    setRefineSaveError(null)
    try {
      await refinement.saveAnnotations(image.image_id, image.filename, effectiveJobId)
      setCurrentStep(4) // Advance to measurement step
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
    setCalibrateError(null)
    setRefineMode(false)
    setDrawMode(false)
    setRefineSaveError(null)
    setOverlayMode('raw')
    setSplitPercent(58)
    setCurrentStep(1)
    setAnnotationPath(false)
  }

  const handleStepNavigation = (targetStep: WorkflowStep) => {
    // Check if navigation is destructive
    const isDestructive =
      (targetStep <= 2 && detectionDone) || (targetStep <= 3 && measurementDone)

    if (isDestructive) {
      setNavigationTarget(targetStep)
      setShowNavigationWarning(true)
    } else {
      setCurrentStep(targetStep)
    }
  }

  const handleConfirmNavigation = () => {
    if (navigationTarget === null) return

    // Clear dependent data
    if (navigationTarget <= 2) {
      // Re-running detection clears everything
      setDetectionJobIdState(null)
      setMeasureJobIdState(null)
      setCsvData(null)
      refinement.loadBoxes([]) // Clear boxes
      setRefineMode(false)
      setAnnotationPath(false)
    } else if (navigationTarget === 3) {
      // Re-editing clears measurements only
      setMeasureJobIdState(null)
      setCsvData(null)
    }

    setCurrentStep(navigationTarget)
    setShowNavigationWarning(false)
    setNavigationTarget(null)
  }

  const handleRowClick = useCallback(
    (rowIndex: number) => {
      const origBoxes = refinement.boxes.filter((b) => b.status !== 'added')
      const box = origBoxes[rowIndex]
      if (!box) return
      if (!refineMode) {
        setRefineMode(true)
        setDrawMode(false)
      }
      refinement.selectBox(box.id)
    },
    [refinement, refineMode]
  )

  const handleImageClick = (x: number, y: number) => {
    // Emit event for manual calibration modal if it's listening
    window.dispatchEvent(
      new CustomEvent('calibration-point-selected', { detail: { x, y } })
    )
  }

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

  // Split drag handler
  const onSplitPointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault()
    const onMove = (ev: PointerEvent) => {
      const rect = mainAreaRef.current?.getBoundingClientRect()
      if (!rect) return
      const pct = ((ev.clientY - rect.top) / rect.height) * 100
      setSplitPercent(Math.max(20, Math.min(80, pct)))
    }
    const onUp = () => {
      document.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerup', onUp)
    }
    document.addEventListener('pointermove', onMove)
    document.addEventListener('pointerup', onUp)
  }, [])

  // ── Step Status Helper ──────────────────────────────────────────────
  const getStepStatus = (step: WorkflowStep): 'pending' | 'active' | 'complete' | 'loading' => {
    if (step === 1) {
      if (currentStep > 1) return 'complete'
      return currentStep === 1 ? 'active' : 'pending'
    }
    if (step === 2) {
      if (detectionDone) return 'complete'
      if (detectionJob && detectionJob.status === 'running') return 'loading'
      return currentStep === 2 ? 'active' : 'pending'
    }
    if (step === 3) {
      if (refinement.annotationsSaved) return 'complete'
      return currentStep === 3 ? 'active' : 'pending'
    }
    if (step === 4) {
      if (measurementDone) return 'complete'
      if (measureJob && measureJob.status === 'running') return 'loading'
      return currentStep === 4 ? 'active' : 'pending'
    }
    return 'pending'
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
          </nav>
          {image && (
            <span
              className="text-xs text-gray-400 truncate max-w-xs hidden sm:block"
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
              {/* Mini workflow dots */}
              <div className="flex flex-col gap-1.5 mt-1">
                {workflowSteps.map((s) => (
                  <div
                    key={s.number}
                    className={`w-2.5 h-2.5 rounded-full ${
                      s.done ? 'bg-green-500' : 'bg-gray-200'
                    }`}
                    title={s.label}
                  />
                ))}
              </div>
            </aside>
          ) : (
            /* Expanded sidebar */
            <aside className="w-80 shrink-0 bg-white border-r flex flex-col overflow-hidden">
              {/* Sidebar header */}
              <div className="px-4 py-3 border-b flex items-center justify-between">
                <h2 className="font-semibold text-gray-900">Workflow</h2>
                <button
                  onClick={() => setSidebarCollapsed(true)}
                  className="text-gray-400 hover:text-gray-700 p-1 rounded hover:bg-gray-100"
                  title="Collapse sidebar"
                >
                  <ChevronRight size={16} className="rotate-180" />
                </button>
              </div>

              {/* Workflow stepper */}
              <div className="px-4 py-4 border-b">
                <div className="flex items-center justify-between">
                  {workflowSteps.map((step, idx) => (
                    <div key={step.number} className="flex items-center">
                      <button
                        onClick={() => handleStepNavigation(step.number as WorkflowStep)}
                        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                          step.done
                            ? 'bg-green-500 text-white'
                            : currentStep === step.number
                            ? 'bg-blue-500 text-white'
                            : 'bg-gray-200 text-gray-600'
                        }`}
                        title={`Step ${step.number}: ${step.label}`}
                      >
                        {step.number}
                      </button>
                      {idx < workflowSteps.length - 1 && (
                        <div
                          className={`w-8 h-0.5 mx-1 ${
                            step.done ? 'bg-green-500' : 'bg-gray-200'
                          }`}
                        />
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Scrollable step content */}
              <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
                {/* Step 1: Set Scale */}
                {currentStep === 1 && (
                  <StepCard
                    stepNumber={1}
                    title="Set Scale"
                    status={getStepStatus(1)}
                  >
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Quick Calibration
                        </label>
                        <div className="flex gap-2">
                          <input
                            type="number"
                            value={umPerPixel}
                            onChange={(e) => setUmPerPixel(parseFloat(e.target.value))}
                            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            step="0.01"
                            placeholder="μm/pixel"
                          />
                          <select
                            value=""
                            onChange={(e) => setUmPerPixel(parseFloat(e.target.value))}
                            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="">Presets</option>
                            {CALIBRATION_PRESETS.map((preset) => (
                              <option key={preset.value} value={preset.value}>
                                {preset.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>

                      <button
                        onClick={() => setModalOpen('manual-calibration')}
                        className="w-full px-4 py-2 border border-blue-300 rounded-lg text-blue-700 hover:bg-blue-50 font-medium transition-colors text-sm"
                      >
                        Manual Setup
                      </button>

                      {calStore.umPerPixel && (
                        <button
                          onClick={() => setCurrentStep(2)}
                          className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium transition-colors text-sm"
                        >
                          Continue to Detection
                        </button>
                      )}

                      {calibrateError && (
                        <p className="text-xs text-red-600">{calibrateError}</p>
                      )}
                    </div>
                  </StepCard>
                )}

                {/* Step 2: Run Detection */}
                {currentStep === 2 && !detectionDone && (
                  <StepCard
                    stepNumber={2}
                    title="Run Detection"
                    status={getStepStatus(2)}
                  >
                    <div className="space-y-3">
                      <button
                        onClick={() => handleRunDetection()}
                        disabled={!calStore.umPerPixel || detectionJob?.status === 'running'}
                        className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium transition-colors text-sm disabled:bg-gray-300 disabled:cursor-not-allowed"
                      >
                        {detectionJob?.status === 'running' ? 'Running...' : 'Run Detection'}
                      </button>

                      <button
                        onClick={() => setModalOpen('advanced-detection')}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors text-sm flex items-center justify-center gap-2"
                      >
                        <Settings2 className="w-4 h-4" />
                        Advanced Settings
                      </button>

                      {detectionJob && (
                        <JobProgress job={detectionJob} />
                      )}

                      {detectionError && (
                        <p className="text-xs text-red-600">{detectionError}</p>
                      )}
                    </div>
                  </StepCard>
                )}

                {/* Step 2: Path Decision (after detection completes) */}
                {currentStep === 2 && detectionDone && (
                  <PathDecisionCard
                    detectionCount={refinement.boxes.length}
                    onSelectPath={handlePathDecision}
                  />
                )}

                {/* Step 3: Edit Detections */}
                {currentStep === 3 && (
                  <StepCard
                    stepNumber={3}
                    title="Edit Detections"
                    status={getStepStatus(3)}
                  >
                    <div className="space-y-3">
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <p className="text-xs text-blue-800">
                          Use the floating toolbar on the viewer to add, remove, or toggle boxes.
                          <br />
                          <br />
                          <strong>Shortcuts:</strong> S (select), D (draw), H (hide/show), Del
                          (remove)
                        </p>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">
                          {refinement.boxes.filter((b) => b.status !== 'rejected').length} boxes
                        </span>
                        <span className="text-gray-600">
                          {refinement.boxes.filter((b) => b.status === 'added').length} added
                        </span>
                      </div>

                      <button
                        onClick={handleDoneEditing}
                        className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium transition-colors text-sm"
                      >
                        Done Editing
                      </button>

                      {role === 'admin' && refinement.annotationsSaved && (
                        <button
                          onClick={() => setModalOpen('finetune')}
                          className="w-full px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 font-medium transition-colors text-sm"
                        >
                          Fine-Tune Model
                        </button>
                      )}

                      {refineSaveError && (
                        <p className="text-xs text-red-600">{refineSaveError}</p>
                      )}
                    </div>
                  </StepCard>
                )}

                {/* Step 4: Measure Organisms */}
                {currentStep === 4 && (
                  <StepCard
                    stepNumber={4}
                    title="Measure Organisms"
                    status={getStepStatus(4)}
                  >
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Measurement Method
                        </label>
                        <div className="space-y-2">
                          <label className="flex items-center gap-2 p-2 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50">
                            <input
                              type="radio"
                              value="fast"
                              checked={measureMethod === 'fast'}
                              onChange={(e) =>
                                setMeasureMethod(e.target.value as 'fast' | 'sam')
                              }
                              className="text-blue-600"
                            />
                            <div className="flex-1">
                              <div className="text-sm font-medium text-gray-900">
                                Fast Ellipse
                              </div>
                              <div className="text-xs text-gray-600">
                                ~178 org/sec, recommended
                              </div>
                            </div>
                          </label>
                          <label className="flex items-center gap-2 p-2 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50">
                            <input
                              type="radio"
                              value="sam"
                              checked={measureMethod === 'sam'}
                              onChange={(e) =>
                                setMeasureMethod(e.target.value as 'fast' | 'sam')
                              }
                              className="text-blue-600"
                            />
                            <div className="flex-1">
                              <div className="text-sm font-medium text-gray-900">
                                SAM Contours
                              </div>
                              <div className="text-xs text-gray-600">
                                ~1 org/sec, more accurate
                              </div>
                            </div>
                          </label>
                        </div>
                      </div>

                      <button
                        onClick={handleRunMeasurement}
                        disabled={
                          !detectionDone ||
                          !calStore.umPerPixel ||
                          measureJob?.status === 'running'
                        }
                        className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium transition-colors text-sm disabled:bg-gray-300 disabled:cursor-not-allowed"
                      >
                        {measureJob?.status === 'running'
                          ? 'Running...'
                          : 'Run Measurement'}
                      </button>

                      {measureJob && <JobProgress job={measureJob} />}

                      {measureError && <p className="text-xs text-red-600">{measureError}</p>}

                      {measurementDone && csvData && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                          <p className="text-xs text-green-800">
                            <strong>Complete!</strong> Measured {csvData.length} organisms.
                          </p>
                        </div>
                      )}
                    </div>
                  </StepCard>
                )}

                {/* Reset button */}
                <button
                  onClick={handleReset}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors text-sm flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Reset Workspace
                </button>
              </div>
            </aside>
          )}

          {/* ── Main Content ── */}
          <div ref={mainAreaRef} className="flex-1 flex flex-col overflow-hidden">
            {/* Viewer Toolbar */}
            <ViewerToolbar
              overlayMode={overlayMode}
              onOverlayChange={setOverlayMode}
              availableOverlays={{
                boxes: detectionDone,
                contours: hasSamOverlay,
              }}
              onExport={handleExport}
            />

            {/* Viewer & Table Split */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Viewer */}
              <div style={{ height: `${splitPercent}%` }} className="relative overflow-hidden">
                <ImageViewer
                  src={viewerSrc}
                  alt={image.filename}
                  className="h-full"
                  onImageClick={handleImageClick}
                  transformOverlay={showBboxOverlay && refineMode ? (
                    <BboxOverlay
                      boxes={refinement.boxes}
                      imageWidth={image.width}
                      imageHeight={image.height}
                      selectedId={refinement.selectedId}
                      onBoxClick={refinement.selectBox}
                      drawingBox={refinement.drawingBox}
                      mode={drawMode ? 'draw' : 'review'}
                      onDrawStart={refinement.startDraw}
                      onDrawMove={refinement.updateDraw}
                      onDrawEnd={refinement.commitDraw}
                    />
                  ) : undefined}
                  disablePan={refineMode && drawMode}
                />

                {/* Floating toolbar for edit mode */}
                {refineMode && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 flex flex-col gap-0.5 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg p-1.5">
                    {(['review', 'draw'] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => {
                          setDrawMode(m === 'draw')
                          refinement.selectBox(null)
                        }}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between gap-3 ${
                          (m === 'draw') === drawMode
                            ? 'bg-amber-500 text-white'
                            : 'text-gray-600 hover:bg-gray-100'
                        }`}
                      >
                        <span>{m === 'review' ? 'Select' : 'Draw'}</span>
                        <kbd className={`text-[9px] px-1 py-0.5 rounded border font-mono ${
                          (m === 'draw') === drawMode
                            ? 'border-amber-300 bg-amber-400/40'
                            : 'border-gray-200 bg-gray-50 text-gray-400'
                        }`}>
                          {m === 'review' ? 'S' : 'D'}
                        </kbd>
                      </button>
                    ))}

                    <div className="border-t border-gray-200 my-0.5" />

                    <button
                      onClick={() => setOverlayMode((prev) => prev === 'raw' ? 'boxes' : 'raw')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between gap-3 ${
                        overlayMode === 'boxes' || overlayMode === 'both'
                          ? 'text-gray-600 hover:bg-gray-100'
                          : 'bg-amber-50 text-amber-700 hover:bg-amber-100'
                      }`}
                    >
                      <span>{overlayMode === 'boxes' || overlayMode === 'both' ? 'Hide' : 'Show'}</span>
                      <kbd className="text-[9px] px-1 py-0.5 rounded border border-gray-200 bg-gray-50 text-gray-400 font-mono">
                        H
                      </kbd>
                    </button>
                  </div>
                )}
              </div>

              {/* Split handle */}
              <div
                onPointerDown={onSplitPointerDown}
                className="h-1 bg-gray-200 cursor-ns-resize hover:bg-blue-400 transition-colors"
              />

              {/* Measurement Table */}
              <div
                style={{ height: `${100 - splitPercent}%` }}
                className="overflow-hidden border-t"
              >
                <MeasurementTable
                  data={csvData ?? []}
                  onRowClick={handleRowClick}
                  selectedIndex={selectedMeasurementIndex}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Modals ── */}
      <ManualCalibrationModal
        isOpen={modalOpen === 'manual-calibration'}
        onClose={() => setModalOpen(null)}
        onComplete={handleManualCalibrationComplete}
        imageWidth={image?.width ?? 0}
        imageHeight={image?.height ?? 0}
      />

      <AdvancedDetectionModal
        isOpen={modalOpen === 'advanced-detection'}
        onClose={() => setModalOpen(null)}
        onSubmit={(config) => {
          handleRunDetection(config)
          setModalOpen(null)
        }}
        initialConfig={{ tileSize, overlap, confidence: conf, iouThreshold: 0.5, device }}
      />

      <FineTuneModal
        isOpen={modalOpen === 'finetune'}
        onClose={() => setModalOpen(null)}
        onSubmit={(config) => {
          // TODO: Implement fine-tuning API call
          console.log('Fine-tune config:', config)
          setModalOpen(null)
        }}
        detectionJobId={detectionJobId ?? undefined}
      />

      <NavigationWarningDialog
        isOpen={showNavigationWarning}
        onClose={() => {
          setShowNavigationWarning(false)
          setNavigationTarget(null)
        }}
        onConfirm={handleConfirmNavigation}
        targetStep={navigationTarget ?? 1}
        hasDetections={detectionDone}
        hasMeasurements={measurementDone}
      />
    </div>
  )
}
