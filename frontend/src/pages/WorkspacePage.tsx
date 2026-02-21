import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { LogOut, Download, RefreshCw } from 'lucide-react'
import ImageUploader from '../components/ImageUploader'
import ImageViewer from '../components/ImageViewer'
import JobProgress from '../components/JobProgress'
import MeasurementTable from '../components/MeasurementTable'
import {
  runDetection,
  runMeasurement,
  autoCalibrate,
  thumbnailUrl,
  outputFileUrl,
} from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import { useCalibrationStore } from '../store/calibrationStore'
import { useAuthStore } from '../store/authStore'
import type { ImageInfo } from '../api/types'

export default function WorkspacePage() {
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const storedUm = useCalibrationStore((s) => s.umPerPixel)
  const setCalibrationStore = useCalibrationStore((s) => s.setCalibration)

  const [image, setImage] = useState<ImageInfo | null>(null)
  const [umPerPixel, setUmPerPixel] = useState<number>(storedUm ?? 8.57)
  const [calibrating, setCalibrating] = useState(false)
  const [calibrateError, setCalibrateError] = useState<string | null>(null)

  const [conf, setConf] = useState(0.6)
  const [detectionJobId, setDetectionJobId] = useState<string | null>(null)
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const detectionJob = useJobProgress(detectionJobId)

  const [measureJobId, setMeasureJobId] = useState<string | null>(null)
  const [measureError, setMeasureError] = useState<string | null>(null)
  const measureJob = useJobProgress(measureJobId)
  const [csvData, setCsvData] = useState<Record<string, any>[] | null>(null)

  const detectionDone = detectionJob?.status === 'completed'
  const measurementDone = measureJob?.status === 'completed'

  const overlayReady = detectionDone && detectionJob?.result?.image_stem
  const viewerSrc = overlayReady
    ? outputFileUrl(detectionJob!.id, `${detectionJob!.result.image_stem}_overlay.jpg`)
    : image
    ? thumbnailUrl(image.image_id)
    : ''

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

  const handleAutoCalibrate = async () => {
    if (!image) return
    setCalibrating(true)
    setCalibrateError(null)
    try {
      const res = await autoCalibrate(image.image_id, 10)
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

  const handleRunDetection = async () => {
    if (!image) return
    setDetectionError(null)
    try {
      const res = await runDetection({ image_id: image.image_id, conf, iou: 0.5, tile_size: 1280, overlap: 256 })
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
        method: 'fast',
      })
      setMeasureJobId(res.job_id)
    } catch (e: any) {
      setMeasureError(e.message)
    }
  }

  const handleReset = () => {
    setImage(null)
    setDetectionJobId(null)
    setMeasureJobId(null)
    setCsvData(null)
    setDetectionError(null)
    setMeasureError(null)
    setCalibrateError(null)
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      {/* Top bar */}
      <header className="shrink-0 bg-white border-b px-5 h-12 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-gray-900">Collembola</span>
          {image && (
            <>
              <span className="text-gray-300">·</span>
              <span className="text-sm text-gray-500 truncate max-w-xs">{image.filename}</span>
              <span className="text-xs text-gray-400">
                {image.width.toLocaleString()}×{image.height.toLocaleString()}px
              </span>
            </>
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
          {/* Left panel */}
          <aside className="w-72 shrink-0 bg-white border-r flex flex-col overflow-y-auto">
            <div className="p-5 space-y-5">

              {/* Image section */}
              <section className="space-y-1">
                <Label>Image</Label>
                <p className="text-sm font-medium text-gray-800 truncate">{image.filename}</p>
                <p className="text-xs text-gray-400">
                  {image.width.toLocaleString()} × {image.height.toLocaleString()} px
                </p>
                <button onClick={handleReset} className="flex items-center gap-1 text-xs text-blue-600 hover:underline mt-0.5">
                  <RefreshCw size={11} /> Load different image
                </button>
              </section>

              <Divider />

              {/* Scale section */}
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
                  />
                  <span className="text-xs text-gray-500">µm / px</span>
                </div>
                <button
                  onClick={handleAutoCalibrate}
                  disabled={calibrating}
                  className="text-xs text-blue-600 hover:underline disabled:opacity-50"
                >
                  {calibrating ? 'Detecting…' : 'Auto-detect from ruler'}
                </button>
                {calibrateError && <p className="text-xs text-amber-600">{calibrateError}</p>}
              </section>

              <Divider />

              {/* Detection section */}
              <section className="space-y-3">
                <Label>Detection</Label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-12">Confidence</span>
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
                  <JobProgress job={detectionJob} />
                )}

                {detectionDone && (
                  <p className="text-sm font-medium text-green-700">
                    ✓ {detectionJob!.result.num_detections} organisms detected
                  </p>
                )}
              </section>

              {/* Measurement section — only appears after detection */}
              {detectionDone && (
                <>
                  <Divider />
                  <section className="space-y-3">
                    <Label>Measurements</Label>

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
                      <JobProgress job={measureJob} />
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

          {/* Main area: image + table */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className={csvData ? 'h-[58%] shrink-0' : 'flex-1'}>
              <ImageViewer src={viewerSrc} alt={image.filename} className="h-full" />
            </div>

            {csvData && (
              <div className="flex-1 overflow-auto border-t bg-white">
                <div className="p-4">
                  <p className="text-xs text-gray-400 mb-3 font-medium uppercase tracking-wide">
                    Measurements — {csvData.length} organisms
                  </p>
                  <MeasurementTable data={csvData} />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{children}</p>
  )
}

function Divider() {
  return <div className="border-t border-gray-100" />
}
