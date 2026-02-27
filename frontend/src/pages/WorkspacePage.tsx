import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogOut, Download, RefreshCw, ChevronDown, ChevronRight, ChevronLeft } from 'lucide-react'
import ImageUploader from '../components/ImageUploader'
import ImageViewer from '../components/ImageViewer'
import BboxOverlay from '../components/BboxOverlay'
import FineTunePanel from '../components/FineTunePanel'
import JobProgress from '../components/JobProgress'
import MeasurementTable from '../components/MeasurementTable'
import {
  runDetection,
  runMeasurement,
  autoCalibrate,
  manualCalibrate,
  imageUrl,
  outputFileUrl,
  annotationExportUrl,
  updateProjectImageJobs,
} from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import { useRefinement } from '../hooks/useRefinement'
import { useCalibrationStore } from '../store/calibrationStore'
import { useAuthStore } from '../store/authStore'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useProjectStore } from '../store/projectStore'
import type { ImageInfo } from '../api/types'

export default function WorkspacePage() {
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const role = useAuthStore((s) => s.role)
  const calStore = useCalibrationStore()
  const workspaceStore = useWorkspaceStore()
  const { currentProjectId, currentProjectName } = useProjectStore()

  // ── Image ──────────────────────────────────────────────────────────
  const [image, setImageState] = useState<ImageInfo | null>(workspaceStore.image)
  const setImage = (img: ImageInfo | null) => {
    setImageState(img)
    workspaceStore.setImage(img)
  }

  // ── Calibration ────────────────────────────────────────────────────
  const [umPerPixel, setUmPerPixelState] = useState<number>(calStore.umPerPixel ?? 8.57)
  const setUmPerPixel = (v: number) => {
    setUmPerPixelState(v)
    calStore.setUmManual(v)
  }
  const rulerMm = calStore.rulerMm
  const setRulerMm = calStore.setRulerMm
  const [calibrating, setCalibrating] = useState(false)
  const [calibrateError, setCalibrateError] = useState<string | null>(null)
  const [calMode, setCalMode] = useState(false)
  const [calPoints, setCalPoints] = useState<[number, number][]>([])

  // ── Detection ──────────────────────────────────────────────────────
  const [conf, setConf] = useState(0.6)
  const [tileSize, setTileSize] = useState(1280)
  const [overlap, setOverlap] = useState(256)
  const [device, setDevice] = useState('0')
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [detectionJobId, setDetectionJobIdState] = useState<string | null>(workspaceStore.detectionJobId)
  const setDetectionJobId = (id: string | null) => {
    setDetectionJobIdState(id)
    workspaceStore.setDetectionJobId(id)
  }
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const detectionJob = useJobProgress(detectionJobId)
  const detectionDone = detectionJob?.status === 'completed'

  // ── Refinement ─────────────────────────────────────────────────────
  const [refineMode, setRefineMode] = useState(false)
  const [drawMode, setDrawMode] = useState(false)
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [refineSaveError, setRefineSaveError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<string | undefined>()
  const refinement = useRefinement(
    image?.image_id ?? null,
    detectionDone ? detectionJobId : null,
  )

  // ── Measurement ────────────────────────────────────────────────────
  const [measureMethod, setMeasureMethod] = useState<'fast' | 'sam'>('fast')
  const [measureJobId, setMeasureJobIdState] = useState<string | null>(workspaceStore.measureJobId)
  const setMeasureJobId = (id: string | null) => {
    setMeasureJobIdState(id)
    workspaceStore.setMeasureJobId(id)
  }
  const [measureError, setMeasureError] = useState<string | null>(null)
  const measureJob = useJobProgress(measureJobId)
  const [csvData, setCsvData] = useState<Record<string, any>[] | null>(null)
  const measurementDone = measureJob?.status === 'completed'

  // ── View state ─────────────────────────────────────────────────────
  const [showContours, setShowContours] = useState(false)
  const [splitPercent, setSplitPercent] = useState(58)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const mainAreaRef = useRef<HTMLDivElement>(null)

  // ── Derived ────────────────────────────────────────────────────────
  const hasSamOverlay = measurementDone && !!measureJob?.result?.overlay_path
  const samOverlayUrl = hasSamOverlay
    ? outputFileUrl(measureJob!.id, measureJob!.result.overlay_path.split('/').pop()!)
    : null

  // Always show raw image; showContours swaps in the SAM overlay JPEG as the base.
  // refineMode adds the SVG BboxOverlay on top — works with either base.
  const viewerSrc = showContours && samOverlayUrl
    ? samOverlayUrl
    : image
    ? imageUrl(image.image_id, image.filename)
    : ''

  const overlayLabel = showContours
    ? 'SAM contours'
    : refineMode
    ? 'Edit mode'
    : null

  // Workflow stepper
  const steps = [
    { label: 'Scale', done: calStore.umPerPixel != null },
    { label: 'Detect', done: detectionDone },
    { label: 'Annotate', done: refinement.annotationsSaved },
    { label: 'Measure', done: measurementDone },
  ]

  // Index of selected box within non-added boxes → maps to measurement CSV row
  const selectedMeasurementIndex = useMemo(() => {
    if (!refinement.selectedId) return null
    const box = refinement.boxes.find((b) => b.id === refinement.selectedId)
    if (!box || box.status === 'added') return null
    const origBoxes = refinement.boxes.filter((b) => b.status !== 'added')
    const idx = origBoxes.findIndex((b) => b.id === refinement.selectedId)
    return idx >= 0 ? idx : null
  }, [refinement.selectedId, refinement.boxes])

  // ── Effects ────────────────────────────────────────────────────────

  // Auto-enter refine mode when saved annotations are restored
  useEffect(() => {
    if (refinement.annotationsSaved && !refineMode) setRefineMode(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refinement.annotationsSaved])

  // Auto-show contours when SAM measurement completes
  useEffect(() => {
    if (hasSamOverlay) setShowContours(true)
  }, [hasSamOverlay])

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
            }),
          )
        })
        .catch(() => {})
    }
  }, [measureJob?.status, measureJob?.result?.csv_path, measureJob?.id])

  // Auto-sync job IDs to project
  useEffect(() => {
    if (!currentProjectId || !image || !detectionJobId) return
    if (detectionJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, { detection_job_id: detectionJobId })
      .catch(() => {})
  }, [detectionJob?.status, detectionJobId, currentProjectId, image?.image_id])

  useEffect(() => {
    if (!currentProjectId || !image || !measureJobId) return
    if (measureJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, { measurement_job_id: measureJobId })
      .catch(() => {})
  }, [measureJob?.status, measureJobId, currentProjectId, image?.image_id])

  // Unified keyboard handler: Esc, S, D, H, Delete/Backspace
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      if (e.key === 'Escape') {
        if (calMode) { setCalMode(false); setCalPoints([]); return }
        if (drawMode) { setDrawMode(false); return }
        if (refinement.selectedId) { refinement.selectBox(null); return }
        return
      }

      if (!refineMode) return

      if (e.key === 's' || e.key === 'S') { setDrawMode(false); refinement.selectBox(null) }
      if (e.key === 'd' || e.key === 'D') { setDrawMode(true); setShowAnnotations(true) }
      if (e.key === 'h' || e.key === 'H') { setShowAnnotations((v) => !v) }

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
  }, [calMode, drawMode, refineMode, refinement])

  // ── Handlers ───────────────────────────────────────────────────────

  const handleAutoCalibrate = async () => {
    if (!image) return
    setCalibrating(true)
    setCalibrateError(null)
    try {
      const res = await autoCalibrate(image.image_id, rulerMm)
      if (res.um_per_pixel) {
        setUmPerPixelState(res.um_per_pixel)
        calStore.setCalibration(res.um_per_pixel, res.calibration_id!, res.method, res.confidence)
      } else {
        setCalibrateError(res.error ?? 'Auto-detect failed — enter value manually')
      }
    } catch (e: any) {
      setCalibrateError(e.message)
    } finally {
      setCalibrating(false)
    }
  }

  const handleImageClick = (x: number, y: number) => {
    if (!calMode) return
    const next = [...calPoints, [x, y]] as [number, number][]
    if (next.length === 2) {
      setCalPoints([])
      setCalMode(false)
      runManualCalibrate(next[0], next[1])
    } else {
      setCalPoints(next)
    }
  }

  const runManualCalibrate = async (p1: [number, number], p2: [number, number]) => {
    if (!image) return
    setCalibrating(true)
    setCalibrateError(null)
    try {
      const res = await manualCalibrate(image.image_id, p1, p2, rulerMm)
      if (res.um_per_pixel) {
        setUmPerPixelState(Number(res.um_per_pixel.toFixed(4)))
        calStore.setCalibration(res.um_per_pixel, res.calibration_id!, res.method, res.confidence)
      }
    } catch (e: any) {
      setCalibrateError(e.message)
    } finally {
      setCalibrating(false)
    }
  }

  const handleRunDetection = async () => {
    if (!image) return
    setDetectionError(null)
    try {
      const res = await runDetection({
        image_id: image.image_id,
        conf,
        iou: 0.5,
        tile_size: tileSize,
        overlap,
        device,
      })
      setDetectionJobId(res.job_id)
    } catch (e: any) {
      setDetectionError(e.message)
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
    setCalMode(false)
    setCalPoints([])
    setRefineMode(false)
    setDrawMode(false)
    setRefineSaveError(null)
    setShowContours(false)
    setSplitPercent(58)
  }

  const handleSaveAnnotations = async () => {
    const effectiveJobId = detectionJobId ?? refinement.restoredSourceJobId
    if (!image || !effectiveJobId) return
    setRefineSaveError(null)
    try {
      await refinement.saveAnnotations(image.image_id, image.filename, effectiveJobId)
    } catch (e: any) {
      setRefineSaveError(e.message)
    }
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
    [refinement, refineMode],
  )

  // Split drag handler — document-level listeners during drag
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

  // ── Render ─────────────────────────────────────────────────────────

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
            <span className="text-xs text-gray-400 truncate max-w-xs hidden sm:block" title={image.filename}>
              {image.filename}
            </span>
          )}
        </div>
        <button
          onClick={() => { logout(); navigate('/login', { replace: true }) }}
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
              <h2 className="text-xl font-semibold text-gray-900">Load an image to get started</h2>
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

          {/* ── Left panel ── */}
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
                {steps.map((s) => (
                  <div
                    key={s.label}
                    className={`w-2.5 h-2.5 rounded-full ${s.done ? 'bg-green-500' : 'bg-gray-200'}`}
                    title={s.label}
                  />
                ))}
              </div>
            </aside>
          ) : (
            /* Full sidebar */
            <aside className="w-72 shrink-0 bg-white border-r flex flex-col overflow-hidden">
              {/* Workflow stepper */}
              <div className="shrink-0 px-4 pt-3 pb-2 border-b bg-gray-50/60">
                <div className="flex items-center">
                  {steps.map((step, i) => (
                    <div key={step.label} className="flex items-center" style={{ flex: i < steps.length - 1 ? '1' : 'none' }}>
                      <div className="flex flex-col items-center">
                        <div
                          className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                            step.done ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-400'
                          }`}
                        >
                          {step.done ? '✓' : i + 1}
                        </div>
                        <span className="text-[10px] text-gray-400 mt-0.5 whitespace-nowrap">{step.label}</span>
                      </div>
                      {i < steps.length - 1 && (
                        <div
                          className={`flex-1 h-px mb-3.5 mx-1 ${step.done ? 'bg-green-300' : 'bg-gray-200'}`}
                        />
                      )}
                    </div>
                  ))}
                  <button
                    onClick={() => setSidebarCollapsed(true)}
                    className="ml-auto text-gray-300 hover:text-gray-500 p-1 rounded hover:bg-gray-100 mb-3.5"
                    title="Collapse sidebar"
                  >
                    <ChevronLeft size={14} />
                  </button>
                </div>
              </div>

              {/* Scrollable content */}
              <div className="flex-1 overflow-y-auto p-5 space-y-5">

                {/* Image */}
                <section className="space-y-1">
                  <Label>Image</Label>
                  <p className="text-sm font-medium text-gray-800 truncate" title={image.filename}>
                    {image.filename}
                  </p>
                  <p className="text-xs text-gray-400">
                    {image.width.toLocaleString()} × {image.height.toLocaleString()} px
                  </p>
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1 text-xs text-blue-600 hover:underline mt-0.5"
                  >
                    <RefreshCw size={11} /> Load different image
                  </button>
                </section>

                <Divider />

                {/* Scale / Calibration */}
                <section className="space-y-2">
                  <Label>Scale</Label>

                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      value={umPerPixel}
                      onChange={(e) => setUmPerPixel(Number(e.target.value))}
                      className="w-24 border rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                      step={0.01}
                      min={0.01}
                      disabled={calibrating}
                    />
                    <span className="text-xs text-gray-500">µm / px</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <label className="text-xs text-gray-500 w-14 shrink-0">Ruler</label>
                    <input
                      type="number"
                      value={rulerMm}
                      onChange={(e) => setRulerMm(Number(e.target.value))}
                      className="w-16 border rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                      step={1}
                      min={0.1}
                    />
                    <span className="text-xs text-gray-500">mm</span>
                  </div>

                  <div className="flex gap-2 flex-wrap">
                    <button
                      onClick={handleAutoCalibrate}
                      disabled={calibrating}
                      className="text-xs px-2.5 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-lg disabled:opacity-50 transition-colors"
                    >
                      {calibrating ? 'Detecting…' : 'Auto-detect'}
                    </button>
                    <button
                      onClick={() => { setCalPoints([]); setCalMode((v) => !v); setCalibrateError(null) }}
                      className={`text-xs px-2.5 py-1.5 rounded-lg transition-colors ${
                        calMode
                          ? 'bg-orange-100 text-orange-700 ring-1 ring-orange-300'
                          : 'bg-gray-100 hover:bg-gray-200'
                      }`}
                    >
                      {calMode ? 'Cancel' : 'Manual'}
                    </button>
                  </div>

                  {calMode && (
                    <div className="rounded-lg bg-orange-50 border border-orange-200 p-2.5 text-xs text-orange-800 space-y-0.5">
                      <p className="font-medium">Click two points on the ruler</p>
                      <p className="text-orange-600">
                        {calPoints.length === 0
                          ? 'Click the first point on the image →'
                          : 'Point 1 captured. Click the second point →'}
                      </p>
                      <p className="text-orange-500 text-[11px]">Press Esc to cancel</p>
                    </div>
                  )}

                  {calibrateError && <p className="text-xs text-amber-600">{calibrateError}</p>}
                </section>

                <Divider />

                {/* Detection */}
                <section className="space-y-3">
                  <Label>Detection</Label>

                  {/* Confidence slider */}
                  <div className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-500">Confidence</span>
                      <span className="text-xs font-medium text-gray-800 tabular-nums">{conf.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      value={conf}
                      onChange={(e) => setConf(Number(e.target.value))}
                      className="w-full accent-blue-600"
                      step={0.05}
                      min={0.1}
                      max={1.0}
                    />
                    <div className="flex justify-between text-[10px] text-gray-300">
                      <span>0.10</span>
                      <span>1.00</span>
                    </div>
                  </div>

                  {/* Advanced toggle */}
                  <button
                    onClick={() => setAdvancedOpen((v) => !v)}
                    className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
                  >
                    {advancedOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    Advanced
                  </button>

                  {advancedOpen && (
                    <div className="space-y-2 pl-1">
                      <Row label="Tile size">
                        <input
                          type="number"
                          value={tileSize}
                          onChange={(e) => setTileSize(Number(e.target.value))}
                          className="w-20 border rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
                          step={64}
                          min={256}
                        />
                        <span className="text-xs text-gray-400">px</span>
                      </Row>
                      <Row label="Overlap">
                        <input
                          type="number"
                          value={overlap}
                          onChange={(e) => setOverlap(Number(e.target.value))}
                          className="w-20 border rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
                          step={32}
                          min={0}
                        />
                        <span className="text-xs text-gray-400">px</span>
                      </Row>
                      <Row label="Device">
                        <input
                          type="text"
                          value={device}
                          onChange={(e) => setDevice(e.target.value)}
                          className="w-20 border rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
                          placeholder="0 / cpu"
                        />
                      </Row>
                    </div>
                  )}

                  {!detectionJob || detectionJob.status === 'failed' ? (
                    <>
                      <button
                        onClick={handleRunDetection}
                        className="w-full py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
                      >
                        Run Detection
                      </button>
                      {detectionError && <p className="text-xs text-red-600">{detectionError}</p>}
                      {detectionJob?.status === 'failed' && detectionJob.error && (
                        <p className="text-xs text-red-600">{detectionJob.error}</p>
                      )}
                    </>
                  ) : (
                    <>
                      <JobProgress job={detectionJob} />
                      {detectionDone && (
                        <button
                          onClick={() => setDetectionJobId(null)}
                          className="text-xs text-gray-400 hover:text-gray-600"
                        >
                          Re-run with different params
                        </button>
                      )}
                    </>
                  )}

                  {detectionDone && (
                    <p className="text-sm font-medium text-green-700">
                      ✓ {detectionJob!.result.num_detections} organisms detected
                    </p>
                  )}

                  {detectionDone && !refineMode && (
                    <button
                      onClick={() => { setRefineMode(true); setDrawMode(false) }}
                      className="w-full py-2 bg-amber-500 text-white rounded-lg text-sm font-medium hover:bg-amber-600 transition-colors"
                    >
                      Edit Detections
                    </button>
                  )}
                </section>

                {/* Refinement panel */}
                {refineMode && (
                  <>
                    <Divider />
                    <section className="space-y-3">
                      <div className="flex items-center justify-between">
                        <Label>Edit Detections</Label>
                        <button
                          onClick={() => { setRefineMode(false); setDrawMode(false) }}
                          className="text-xs text-gray-400 hover:text-gray-600"
                        >
                          Exit
                        </button>
                      </div>

                      {refinement.isLoading && <p className="text-xs text-gray-500">Loading boxes…</p>}
                      {refinement.error && <p className="text-xs text-red-600">{refinement.error}</p>}

                      {!refinement.isLoading && (
                        <p className="text-xs text-gray-600">
                          <span className="text-green-600 font-medium">✓ {refinement.acceptedCount}</span>
                          {refinement.addedCount > 0 && (
                            <span className="text-blue-500 font-medium"> · +{refinement.addedCount} drawn</span>
                          )}
                          {refinement.rejectedCount > 0 && (
                            <span className="text-red-500 font-medium"> · ✗ {refinement.rejectedCount} invalid</span>
                          )}
                        </p>
                      )}

                      <button
                        onClick={handleSaveAnnotations}
                        disabled={refinement.isSaving}
                        className="w-full py-2 bg-gray-800 text-white rounded-lg text-sm font-medium hover:bg-gray-900 disabled:opacity-50 transition-colors"
                      >
                        {refinement.isSaving ? 'Saving…' : 'Save Annotations'}
                      </button>
                      {refineSaveError && <p className="text-xs text-red-600">{refineSaveError}</p>}
                      {refinement.annotationsSaved && (
                        <p className="text-xs text-green-600">✓ Annotations saved</p>
                      )}

                      {refinement.annotationsSaved && image && (
                        <div className="flex gap-2">
                          <a
                            href={annotationExportUrl(image.image_id, 'json')}
                            download
                            className="flex-1 text-center text-xs py-1.5 border rounded-lg hover:bg-gray-50 transition-colors"
                          >
                            JSON
                          </a>
                          <a
                            href={annotationExportUrl(image.image_id, 'csv')}
                            download
                            className="flex-1 text-center text-xs py-1.5 border rounded-lg hover:bg-gray-50 transition-colors"
                          >
                            CSV
                          </a>
                        </div>
                      )}

                      {role === 'admin' && refinement.annotationsSaved && image && detectionJobId && (
                        <>
                          <Divider />
                          <Label>Fine-tune</Label>
                          <FineTunePanel
                            imageId={image.image_id}
                            onModelSelected={(path) => setSelectedModel(path)}
                          />
                          {selectedModel && (
                            <p className="text-xs text-violet-700 truncate">
                              Active: {selectedModel.split('/').pop()}
                            </p>
                          )}
                        </>
                      )}
                    </section>
                  </>
                )}

                {/* Project context */}
                {detectionDone && currentProjectId && (
                  <>
                    <Divider />
                    <section className="space-y-1.5">
                      <Label>Project</Label>
                      <p className="text-xs text-green-700 bg-green-50 rounded px-2 py-1.5">
                        ✓ Results saved to{' '}
                        <Link to={`/projects/${currentProjectId}`} className="font-medium hover:underline">
                          {currentProjectName ?? 'project'}
                        </Link>
                      </p>
                    </section>
                  </>
                )}

                {/* Measurements */}
                {detectionDone && (
                  <>
                    <Divider />
                    <section className="space-y-3">
                      <Label>Measurements</Label>

                      <div className="flex gap-1.5">
                        {(['fast', 'sam'] as const).map((m) => (
                          <button
                            key={m}
                            onClick={() => setMeasureMethod(m)}
                            className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                              measureMethod === m
                                ? 'bg-blue-50 border-blue-300 text-blue-700 font-medium'
                                : 'bg-white border-gray-200 text-gray-500 hover:border-gray-300'
                            }`}
                          >
                            {m === 'fast' ? 'Fast (ellipse)' : 'Accurate (SAM)'}
                          </button>
                        ))}
                      </div>
                      {measureMethod === 'sam' && (
                        <p className="text-xs text-amber-600">
                          SAM is ~1 org/sec — may take several minutes for large samples.
                        </p>
                      )}

                      {refinement.annotationsSaved ? (
                        <p className="text-xs text-teal-700 bg-teal-50 rounded px-2 py-1.5">
                          ✓ Using your edited detections
                        </p>
                      ) : refinement.boxes.some((b) => b.status !== 'accepted') ? (
                        <p className="text-xs text-amber-600">
                          Save annotations first to include your edits.
                        </p>
                      ) : null}

                      {!measureJob || measureJob.status === 'failed' ? (
                        <>
                          <button
                            onClick={handleRunMeasurement}
                            className="w-full py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 transition-colors"
                          >
                            Measure Organisms
                          </button>
                          {measureError && <p className="text-xs text-red-600">{measureError}</p>}
                          {measureJob?.status === 'failed' && measureJob.error && (
                            <p className="text-xs text-red-600">{measureJob.error}</p>
                          )}
                        </>
                      ) : (
                        <>
                          <JobProgress job={measureJob} />
                          {measurementDone && (
                            <button
                              onClick={() => { setMeasureJobId(null); setCsvData(null) }}
                              className="text-xs text-gray-400 hover:text-gray-600"
                            >
                              Re-run measurements
                            </button>
                          )}
                        </>
                      )}

                      {measurementDone && (
                        <>
                          <p className="text-sm font-medium text-green-700">
                            ✓ {measureJob!.result.num_organisms} organisms measured
                          </p>
                          <a
                            href={outputFileUrl(
                              measureJob!.id,
                              measureJob!.result.csv_path.split('/').pop()!,
                            )}
                            download
                            className="flex items-center justify-center gap-2 w-full py-2 border border-gray-300 rounded-lg text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                          >
                            <Download size={14} />
                            Export CSV
                          </a>
                        </>
                      )}
                    </section>
                  </>
                )}
              </div>
            </aside>
          )}

          {/* ── Main area: viewer + table ── */}
          <div className="flex-1 flex flex-col overflow-hidden" ref={mainAreaRef}>

            {/* Viewer */}
            <div
              className="relative shrink-0"
              style={{ height: csvData ? `calc(${splitPercent}% - 3px)` : '100%' }}
            >
              {/* Contours toggle — only visible when SAM overlay exists */}
              {hasSamOverlay && (
                <button
                  onClick={() => setShowContours((v) => !v)}
                  className={`absolute top-2 left-2 z-20 text-xs px-3 py-1.5 rounded-lg border shadow-sm transition-colors ${
                    showContours
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white/90 backdrop-blur-sm border-gray-200 text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  Contours
                </button>
              )}

              {/* Overlay label badge — only when there's something to say */}
              {overlayLabel && (
                <div className="absolute bottom-2 left-2 z-20 pointer-events-none">
                  <span className="text-[11px] bg-black/40 text-white rounded px-2 py-0.5 backdrop-blur-sm">
                    {overlayLabel}
                  </span>
                </div>
              )}

              {/* Floating annotation toolbar */}
              {refineMode && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 flex flex-col gap-0.5 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg p-1.5">
                  {(['review', 'draw'] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => {
                        setDrawMode(m === 'draw')
                        refinement.selectBox(null)
                        if (m === 'draw') setShowAnnotations(true)
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
                    onClick={() => setShowAnnotations((v) => !v)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between gap-3 ${
                      showAnnotations
                        ? 'text-gray-600 hover:bg-gray-100'
                        : 'bg-amber-50 text-amber-700 hover:bg-amber-100'
                    }`}
                  >
                    <span>{showAnnotations ? 'Hide' : 'Show'}</span>
                    <kbd className="text-[9px] px-1 py-0.5 rounded border border-gray-200 bg-gray-50 text-gray-400 font-mono">
                      H
                    </kbd>
                  </button>

                  {!drawMode && refinement.selectedId && (() => {
                    const sel = refinement.boxes.find((b) => b.id === refinement.selectedId)
                    const isAdded = sel?.status === 'added'
                    const isRejected = sel?.status === 'rejected'
                    return (
                      <>
                        <div className="border-t border-gray-200 my-0.5" />
                        {isAdded ? (
                          <button
                            onClick={() => refinement.removeBox(refinement.selectedId!)}
                            className="px-3 py-1.5 rounded-lg text-xs font-medium text-red-600 hover:bg-red-50 flex items-center justify-between gap-3"
                          >
                            <span>Delete</span>
                            <kbd className="text-[9px] px-1 py-0.5 rounded border border-gray-200 bg-gray-50 text-gray-400 font-mono">Del</kbd>
                          </button>
                        ) : (
                          <button
                            onClick={() => refinement.toggleBox(refinement.selectedId!)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center justify-between gap-3 ${
                              isRejected
                                ? 'text-green-600 hover:bg-green-50'
                                : 'text-red-600 hover:bg-red-50'
                            }`}
                          >
                            <span>{isRejected ? 'Restore' : 'Invalid'}</span>
                            <kbd className="text-[9px] px-1 py-0.5 rounded border border-gray-200 bg-gray-50 text-gray-400 font-mono">Del</kbd>
                          </button>
                        )}
                        <button
                          onClick={() => refinement.selectBox(null)}
                          className="px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:bg-gray-100 flex items-center justify-between gap-3"
                        >
                          <span>Deselect</span>
                          <kbd className="text-[9px] px-1 py-0.5 rounded border border-gray-200 bg-gray-50 text-gray-400 font-mono">Esc</kbd>
                        </button>
                      </>
                    )
                  })()}
                </div>
              )}

              <ImageViewer
                src={viewerSrc}
                alt={image.filename}
                className="h-full"
                onImageClick={calMode ? handleImageClick : undefined}
                transformOverlay={refineMode && showAnnotations ? (
                  <BboxOverlay
                    boxes={refinement.boxes}
                    imageWidth={image.width}
                    imageHeight={image.height}
                    onBoxClick={refinement.selectBox}
                    selectedId={refinement.selectedId}
                    drawingBox={refinement.drawingBox}
                    mode={drawMode ? 'draw' : 'review'}
                    onDrawStart={refinement.startDraw}
                    onDrawMove={refinement.updateDraw}
                    onDrawEnd={refinement.commitDraw}
                  />
                ) : undefined}
                disablePan={refineMode && drawMode}
              />
            </div>

            {/* Drag handle */}
            {csvData && (
              <div
                className="h-1.5 shrink-0 bg-gray-200 hover:bg-blue-400 active:bg-blue-500 cursor-row-resize transition-colors"
                onPointerDown={onSplitPointerDown}
              />
            )}

            {/* Measurement table */}
            {csvData && (
              <div className="flex-1 flex flex-col overflow-hidden bg-white border-t min-h-0">
                {/* Summary bar */}
                {measureJob?.result?.summary && (
                  <div className="shrink-0 flex gap-5 px-4 py-2 border-b bg-gray-50 overflow-x-auto">
                    {(['length_mm', 'width_mm', 'area_mm2', 'volume_mm3'] as const).map((col) => {
                      const s = measureJob.result.summary[col]
                      if (!s) return null
                      const label = col === 'length_mm' ? 'Length' : col === 'width_mm' ? 'Width' : col === 'area_mm2' ? 'Area' : 'Volume'
                      const unit = col.endsWith('mm3') ? 'mm³' : col.endsWith('mm2') ? 'mm²' : 'mm'
                      return (
                        <div key={col} className="shrink-0 space-y-0.5">
                          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide">
                            {label} ({unit})
                          </p>
                          <p className="text-sm font-medium text-gray-800">
                            {s.mean.toFixed(3)}
                          </p>
                          <p className="text-[10px] text-gray-400">
                            {s.min.toFixed(3)} – {s.max.toFixed(3)}
                          </p>
                        </div>
                      )
                    })}
                    <div className="ml-auto shrink-0 self-center text-xs text-gray-400">
                      n = {csvData.length}
                    </div>
                  </div>
                )}
                <div className="flex-1 overflow-hidden min-h-0">
                  <MeasurementTable
                    data={csvData}
                    selectedIndex={selectedMeasurementIndex}
                    onRowClick={handleRowClick}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Small helpers ──────────────────────────────────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{children}</p>
  )
}

function Divider() {
  return <div className="border-t border-gray-100" />
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500 w-14 shrink-0">{label}</span>
      {children}
    </div>
  )
}
