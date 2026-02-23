import { useState, useEffect, useMemo } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogOut, Download, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react'
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
  submitToCommunity,
} from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import { useRefinement } from '../hooks/useRefinement'
import { useCalibrationStore } from '../store/calibrationStore'
import { useAuthStore } from '../store/authStore'
import { useCommunityLoadStore } from '../store/communityLoadStore'
import { useWorkspaceStore } from '../store/workspaceStore'
import type { ImageInfo } from '../api/types'

export default function WorkspacePage() {
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const role = useAuthStore((s) => s.role)
  const storedUm = useCalibrationStore((s) => s.umPerPixel)
  const setCalibrationStore = useCalibrationStore((s) => s.setCalibration)
  const communityLoad = useCommunityLoadStore()
  const workspaceStore = useWorkspaceStore()

  // ── Image ──────────────────────────────────────────────────────────
  const [image, setImageState] = useState<ImageInfo | null>(workspaceStore.image)
  const setImage = (img: ImageInfo | null) => {
    setImageState(img)
    workspaceStore.setImage(img)
  }

  // ── Calibration ────────────────────────────────────────────────────
  const [umPerPixel, setUmPerPixel] = useState<number>(storedUm ?? 8.57)
  const [rulerMm, setRulerMm] = useState(10)
  const [calibrating, setCalibrating] = useState(false)
  const [calibrateError, setCalibrateError] = useState<string | null>(null)
  // Manual click calibration: collect two points in image coordinates
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
  const [refineSaveError, setRefineSaveError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<string | undefined>()
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'submitting' | 'ok' | 'error'>('idle')
  const [submitError, setSubmitError] = useState<string | null>(null)
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

  // Index of selected box within original (non-added) boxes → maps to measurement CSV row
  const selectedMeasurementIndex = useMemo(() => {
    if (!refinement.selectedId) return null
    const box = refinement.boxes.find((b) => b.id === refinement.selectedId)
    if (!box || box.status === 'added') return null
    const origBoxes = refinement.boxes.filter((b) => b.status !== 'added')
    const idx = origBoxes.findIndex((b) => b.id === refinement.selectedId)
    return idx >= 0 ? idx : null
  }, [refinement.selectedId, refinement.boxes])

  // Keyboard Delete/Backspace removes the selected box when in review mode
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!refineMode || drawMode) return
      if (!refinement.selectedId) return
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        refinement.toggleBox(refinement.selectedId)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [refineMode, drawMode, refinement.selectedId, refinement.removeBox])

  const overlayReady = detectionDone && detectionJob?.result?.image_stem
  // In refine mode always show the original image (SVG overlay is drawn on top)
  const viewerSrc = refineMode
    ? (image ? imageUrl(image.image_id, image.filename) : '')
    : overlayReady
    ? outputFileUrl(detectionJob!.id, `${detectionJob!.result.image_stem}_overlay.jpg`)
    : image
    ? imageUrl(image.image_id, image.filename)
    : ''

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

  // ── Handlers ───────────────────────────────────────────────────────

  const handleAutoCalibrate = async () => {
    if (!image) return
    setCalibrating(true)
    setCalibrateError(null)
    try {
      const res = await autoCalibrate(image.image_id, rulerMm)
      if (res.um_per_pixel) {
        setUmPerPixel(res.um_per_pixel)
        setCalibrationStore(res.um_per_pixel, res.calibration_id!, res.method, res.confidence)
      } else {
        setCalibrateError(res.error ?? 'Auto-detect failed — enter value manually')
      }
    } catch (e: any) {
      setCalibrateError(e.message)
    } finally {
      setCalibrating(false)
    }
  }

  // Called by ImageViewer when user clicks the image in calMode
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
        setUmPerPixel(Number(res.um_per_pixel.toFixed(4)))
        setCalibrationStore(res.um_per_pixel, res.calibration_id!, res.method, res.confidence)
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
    setSubmitStatus('idle')
    setSubmitError(null)
    communityLoad.clear()
  }

  const handleSaveAnnotations = async () => {
    if (!image || !detectionJobId) return
    setRefineSaveError(null)
    try {
      await refinement.saveAnnotations(image.image_id, image.filename, detectionJobId)
    } catch (e: any) {
      setRefineSaveError(e.message)
    }
  }

  const handleSubmitCommunity = async () => {
    if (!image) return
    setSubmitStatus('submitting')
    setSubmitError(null)
    try {
      await submitToCommunity({
        image_name: image.filename,
        image_width: image.width,
        image_height: image.height,
        um_per_pixel: umPerPixel,
        conf_threshold: conf,
        boxes: refinement.boxes.filter((b) => b.status !== 'rejected'),
      })
      setSubmitStatus('ok')
    } catch (e: any) {
      setSubmitStatus('error')
      setSubmitError(e.message)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      {/* Top bar */}
      <header className="shrink-0 bg-white border-b px-5 h-12 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <span className="font-semibold text-gray-900">Collembola</span>
          <nav className="flex items-center gap-1">
            <Link
              to="/"
              className="text-sm px-3 py-1 rounded-md bg-gray-100 text-gray-900 font-medium transition-colors"
            >
              Workspace
            </Link>
            <Link
              to="/community"
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Community
            </Link>
          </nav>
          {image && (
            <span className="text-xs text-gray-400 truncate max-w-xs hidden sm:block">
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
            {communityLoad.numDetections > 0 && (
              <div className="rounded-lg bg-teal-50 border border-teal-200 p-4 text-center space-y-1">
                <p className="text-sm font-medium text-teal-800">
                  📦 {communityLoad.numDetections} boxes ready from community
                </p>
                <p className="text-xs text-teal-600">
                  by <span className="font-medium">{communityLoad.submittedBy}</span>
                  {' · '}{communityLoad.imageName}
                </p>
                <p className="text-xs text-teal-500">Upload the same image below to view them overlaid.</p>
              </div>
            )}
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
          <aside className="w-72 shrink-0 bg-white border-r flex flex-col overflow-y-auto">
            <div className="p-5 space-y-5">

              {/* Image */}
              <section className="space-y-1">
                <Label>Image</Label>
                <p className="text-sm font-medium text-gray-800 truncate">{image.filename}</p>
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

              {/* Community load banner */}
              {communityLoad.numDetections > 0 && (
                <>
                  <Divider />
                  <section className="rounded-lg bg-teal-50 border border-teal-200 p-3 space-y-2">
                    <p className="text-xs font-medium text-teal-800">
                      📦 {communityLoad.numDetections} boxes from{' '}
                      <span className="font-semibold">{communityLoad.submittedBy}</span>
                    </p>
                    <p className="text-xs text-teal-600 truncate" title={communityLoad.imageName}>
                      {communityLoad.imageName}
                    </p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          refinement.loadBoxes(communityLoad.boxes)
                          communityLoad.clear()
                          setRefineMode(true)
                          setDrawMode(false)
                        }}
                        className="flex-1 py-1.5 bg-teal-600 text-white rounded-md text-xs font-medium hover:bg-teal-700 transition-colors"
                      >
                        Apply boxes
                      </button>
                      <button
                        onClick={communityLoad.clear}
                        className="py-1.5 px-2 text-xs text-teal-600 hover:text-teal-800"
                      >
                        Dismiss
                      </button>
                    </div>
                  </section>
                </>
              )}

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

                {/* Ruler length — used by both auto and manual */}
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
                    onClick={() => {
                      setCalPoints([])
                      setCalMode((v) => !v)
                      setCalibrateError(null)
                    }}
                    className={`text-xs px-2.5 py-1.5 rounded-lg transition-colors ${
                      calMode
                        ? 'bg-orange-100 text-orange-700 ring-1 ring-orange-300'
                        : 'bg-gray-100 hover:bg-gray-200'
                    }`}
                  >
                    {calMode ? 'Cancel' : 'Manual'}
                  </button>
                </div>

                {/* Manual calibration guidance */}
                {calMode && (
                  <div className="rounded-lg bg-orange-50 border border-orange-200 p-2.5 text-xs text-orange-800 space-y-0.5">
                    <p className="font-medium">Click two points on the ruler</p>
                    <p className="text-orange-600">
                      {calPoints.length === 0
                        ? 'Click the first point on the image →'
                        : 'Point 1 captured. Click the second point →'}
                    </p>
                  </div>
                )}

                {calibrateError && (
                  <p className="text-xs text-amber-600">{calibrateError}</p>
                )}
              </section>

              <Divider />

              {/* Detection */}
              <section className="space-y-3">
                <Label>Detection</Label>

                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-14 shrink-0">Confidence</span>
                  <input
                    type="number"
                    value={conf}
                    onChange={(e) => setConf(Number(e.target.value))}
                    className="w-20 border rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                    step={0.05}
                    min={0.1}
                    max={1.0}
                  />
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
                    {detectionError && (
                      <p className="text-xs text-red-600">{detectionError}</p>
                    )}
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

                {/* Edit Detections button */}
                {detectionDone && !refineMode && (
                  <button
                    onClick={() => { setRefineMode(true); setDrawMode(false) }}
                    className="w-full py-2 bg-amber-500 text-white rounded-lg text-sm font-medium hover:bg-amber-600 transition-colors"
                  >
                    Edit Detections
                  </button>
                )}
              </section>

              {/* Refinement panel — shown when refineMode is active */}
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

                    {refinement.isLoading && (
                      <p className="text-xs text-gray-500">Loading boxes…</p>
                    )}
                    {refinement.error && (
                      <p className="text-xs text-red-600">{refinement.error}</p>
                    )}

                    {/* Stats */}
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

                    {/* Mode toggle */}
                    <div className="flex gap-1.5">
                      {(['review', 'draw'] as const).map((m) => (
                        <button
                          key={m}
                          onClick={() => {
                            setDrawMode(m === 'draw')
                            refinement.selectBox(null)
                          }}
                          className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                            (m === 'draw') === drawMode
                              ? 'bg-amber-50 border-amber-300 text-amber-700 font-medium'
                              : 'bg-white border-gray-200 text-gray-500 hover:border-gray-300'
                          }`}
                        >
                          {m === 'review' ? 'Select' : 'Draw'}
                        </button>
                      ))}
                    </div>
                    {drawMode && (
                      <p className="text-xs text-amber-700 bg-amber-50 rounded p-2">
                        Drag on the image to draw a new box. Pan is disabled.
                      </p>
                    )}
                    {!drawMode && !refinement.selectedId && (
                      <p className="text-xs text-gray-500">
                        Click a box to select it, then use the badge to mark it invalid or restore it.
                      </p>
                    )}
                    {!drawMode && refinement.selectedId && (() => {
                      const sel = refinement.boxes.find((b) => b.id === refinement.selectedId)
                      const isRejected = sel?.status === 'rejected'
                      return (
                        <div className={`flex items-center gap-2 rounded-lg px-2.5 py-2 border ${
                          isRejected
                            ? 'bg-red-50 border-red-200'
                            : 'bg-amber-50 border-amber-200'
                        }`}>
                          <span className={`text-xs flex-1 ${isRejected ? 'text-red-700' : 'text-amber-700'}`}>
                            {isRejected ? '✗ Marked invalid' : '1 box selected'}
                          </span>
                          <button
                            onClick={() => refinement.toggleBox(refinement.selectedId!)}
                            className={`text-xs font-medium ${
                              isRejected
                                ? 'text-green-600 hover:text-green-800'
                                : 'text-red-600 hover:text-red-800'
                            }`}
                          >
                            {isRejected ? 'Restore' : 'Mark invalid'}
                          </button>
                          <button
                            onClick={() => refinement.selectBox(null)}
                            className="text-xs text-gray-400 hover:text-gray-600"
                          >
                            ✕
                          </button>
                        </div>
                      )
                    })()}

                    {/* Save */}
                    <button
                      onClick={handleSaveAnnotations}
                      disabled={refinement.isSaving}
                      className="w-full py-2 bg-gray-800 text-white rounded-lg text-sm font-medium hover:bg-gray-900 disabled:opacity-50 transition-colors"
                    >
                      {refinement.isSaving ? 'Saving…' : 'Save Annotations'}
                    </button>
                    {refineSaveError && (
                      <p className="text-xs text-red-600">{refineSaveError}</p>
                    )}
                    {refinement.annotationsSaved && (
                      <p className="text-xs text-green-600">✓ Annotations saved</p>
                    )}

                    {/* Export links */}
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

                    {/* Fine-tune — admin only, unlocks after annotations saved */}
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

              {/* Community submit — unlocks after detection */}
              {detectionDone && (
                <>
                  <Divider />
                  <section className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Community</Label>
                      <Link
                        to="/community"
                        className="text-xs text-blue-500 hover:underline"
                      >
                        Browse →
                      </Link>
                    </div>
                    <button
                      onClick={handleSubmitCommunity}
                      disabled={submitStatus === 'submitting' || submitStatus === 'ok'}
                      className="w-full py-2 bg-teal-600 text-white rounded-lg text-sm font-medium hover:bg-teal-700 disabled:opacity-50 transition-colors"
                    >
                      {submitStatus === 'submitting'
                        ? 'Submitting…'
                        : submitStatus === 'ok'
                        ? '✓ Submitted!'
                        : `Submit ${refinement.boxes.filter((b) => b.status !== 'rejected').length} detections`}
                    </button>
                    {submitStatus === 'error' && submitError && (
                      <p className="text-xs text-red-600">{submitError}</p>
                    )}
                    {submitStatus === 'ok' && (
                      <Link
                        to="/community"
                        className="block text-center text-xs text-teal-700 hover:underline"
                      >
                        View in Community →
                      </Link>
                    )}
                  </section>
                </>
              )}

              {/* Measurements — unlocks after detection */}
              {detectionDone && (
                <>
                  <Divider />
                  <section className="space-y-3">
                    <Label>Measurements</Label>

                    {/* Method toggle */}
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
                        ✓ Using your edited detections (kept + added boxes)
                      </p>
                    ) : refinement.boxes.some((b) => b.status !== 'accepted') && (
                      <p className="text-xs text-amber-600">
                        Save annotations first to include your edits in measurements.
                      </p>
                    )}

                    {!measureJob || measureJob.status === 'failed' ? (
                      <>
                        <button
                          onClick={handleRunMeasurement}
                          className="w-full py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 transition-colors"
                        >
                          Measure Organisms
                        </button>
                        {measureError && (
                          <p className="text-xs text-red-600">{measureError}</p>
                        )}
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

          {/* ── Main area: image viewer + measurement table ── */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className={csvData ? 'h-[58%] shrink-0' : 'flex-1'}>
              <ImageViewer
                src={viewerSrc}
                alt={image.filename}
                className="h-full"
                onImageClick={calMode ? handleImageClick : undefined}
                transformOverlay={refineMode ? (
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

            {csvData && (
              <div className="flex-1 overflow-auto border-t bg-white">
                <div className="p-4">
                  <p className="text-xs text-gray-400 mb-3 font-medium uppercase tracking-wide">
                    Measurements — {csvData.length} organisms
                  </p>
                  <MeasurementTable data={csvData} selectedIndex={selectedMeasurementIndex} />
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
