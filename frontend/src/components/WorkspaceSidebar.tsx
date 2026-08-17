import { Settings2, RefreshCw, Table2, CheckCircle2, ImagePlus, Ruler, X } from 'lucide-react'
import JobProgress from './JobProgress'
import type { Job } from '../api/types'

const CALIBRATION_PRESETS = [
  { label: '8.57 μm/px (default)', value: 8.57 },
  { label: '10.0 μm/px', value: 10.0 },
  { label: '5.0 μm/px', value: 5.0 },
]

interface Props {
  // Image
  imageFilename: string
  imageWidth: number
  imageHeight: number

  // Calibration
  umPerPixel: number
  setUmPerPixel: (v: number) => void
  calibrated: boolean
  calibrationMode: boolean
  calibrationPointCount: number
  calibrationError: string | null
  calibrationMessage: string | null
  onStartCalibration: () => void
  onCancelCalibration: () => void

  // Detection
  detectionDone: boolean
  detectionJob: Job | null
  detectionError: string | null
  onRunDetection: () => void
  onAdvancedSettings: () => void

  // Editing (status only — controls are in the floating toolbar)
  annotationsSaved: boolean
  boxCount: number

  // Measurement
  measurementDone: boolean
  measureJob: Job | null
  measureError: string | null
  measureMethod: 'fast' | 'sam'
  setMeasureMethod: (m: 'fast' | 'sam') => void
  onRunMeasurement: () => void
  csvDataLength: number
  onViewResults: () => void

  // Reset
  onReset: () => void
}

export function WorkspaceSidebar({
  imageFilename,
  imageWidth,
  imageHeight,
  umPerPixel,
  setUmPerPixel,
  calibrated,
  calibrationMode,
  calibrationPointCount,
  calibrationError,
  calibrationMessage,
  onStartCalibration,
  onCancelCalibration,
  detectionDone,
  detectionJob,
  detectionError,
  onRunDetection,
  onAdvancedSettings,
  annotationsSaved,
  boxCount,
  measurementDone,
  measureJob,
  measureError,
  measureMethod,
  setMeasureMethod,
  onRunMeasurement,
  csvDataLength,
  onViewResults,
  onReset,
}: Props) {
  const statusDots = [
    { label: 'Calibrated', done: calibrated },
    { label: 'Detected', done: detectionDone },
    { label: 'Annotated', done: annotationsSaved },
    { label: 'Measured', done: measurementDone },
  ]

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Image info */}
      <div className="px-4 pt-4 pb-3 border-b shrink-0">
        <div className="flex items-start justify-between gap-1">
          <p className="text-xs font-medium text-gray-700 truncate min-w-0" title={imageFilename}>
            {imageFilename}
          </p>
          <button
            onClick={onReset}
            title="Upload a different image"
            className="shrink-0 text-gray-400 hover:text-blue-600 transition-colors"
          >
            <ImagePlus size={14} />
          </button>
        </div>
        <p className="text-[10px] text-gray-400 mt-0.5">
          {imageWidth} × {imageHeight} px
        </p>
        <div className="flex items-center gap-3 mt-2.5">
          {statusDots.map((dot) => (
            <div key={dot.label} className="flex flex-col items-center gap-0.5">
              <div
                className={`w-2 h-2 rounded-full ${dot.done ? 'bg-green-500' : 'bg-gray-200'}`}
                title={dot.label}
              />
              <span className="text-[9px] text-gray-400 leading-none">{dot.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 min-h-0 overflow-y-auto">

        {/* ── Scale ───────────────────────────────────────────── */}
        <div className="px-4 py-4 border-b">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Scale</p>
          <input
            type="number"
            value={umPerPixel}
            onChange={(e) => setUmPerPixel(parseFloat(e.target.value))}
            className="w-full px-2.5 py-1.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            step="0.01"
            placeholder="μm/pixel"
          />
          <div className="flex gap-2 mt-1.5">
            <select
              value=""
              onChange={(e) => setUmPerPixel(parseFloat(e.target.value))}
              className="w-full px-2 py-1 border border-gray-300 rounded-lg text-xs bg-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Quick presets…</option>
              {CALIBRATION_PRESETS.map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>
          <button
            onClick={calibrationMode ? onCancelCalibration : onStartCalibration}
            className={`w-full mt-2 px-3 py-1.5 rounded-lg text-sm flex items-center justify-center gap-2 transition-colors ${
              calibrationMode
                ? 'border border-amber-300 text-amber-700 bg-amber-50 hover:bg-amber-100'
                : 'border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            {calibrationMode ? <X size={14} /> : <Ruler size={14} />}
            {calibrationMode ? 'Cancel Calibration' : 'Calibrate from Scale Bar'}
          </button>
          {calibrationMode && (
            <p className="text-[11px] text-amber-700 mt-1.5">
              Click the 0 mm and 10 mm marks on the image. Points selected: {calibrationPointCount}/2.
            </p>
          )}
          {calibrationMessage && <p className="text-[11px] text-green-700 mt-1.5">{calibrationMessage}</p>}
          {calibrationError && <p className="text-[11px] text-red-600 mt-1.5">{calibrationError}</p>}
        </div>

        {/* ── Detection ───────────────────────────────────────── */}
        <div className="px-4 py-4 border-b space-y-2">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Detection</p>

          {detectionDone && (
            <div className="flex items-center gap-1.5 text-xs text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-2">
              <CheckCircle2 size={13} className="shrink-0" />
              {boxCount} detections found
            </div>
          )}

          {detectionJob?.status === 'running' && <JobProgress job={detectionJob} />}

          <button
            onClick={onRunDetection}
            disabled={detectionJob?.status === 'running'}
            className={`w-full px-4 py-2 rounded-lg font-medium transition-colors text-sm disabled:bg-gray-300 disabled:cursor-not-allowed ${
              detectionDone
                ? 'border border-gray-300 text-gray-700 hover:bg-gray-50'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {detectionJob?.status === 'running'
              ? 'Running…'
              : detectionDone
              ? 'Re-run Detection'
              : 'Run Detection'}
          </button>

          <button
            onClick={onAdvancedSettings}
            className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors text-sm flex items-center justify-center gap-2"
          >
            <Settings2 size={14} />
            Advanced Settings
          </button>

          {detectionError && <p className="text-xs text-red-600">{detectionError}</p>}
        </div>

        {/* ── Measurement ─────────────────────────────────────── */}
        {detectionDone && (
          <div className="px-4 py-4 space-y-2">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Measurement</p>

            <div className="space-y-1.5">
              <label className="flex items-center gap-2 p-2 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="radio"
                  value="fast"
                  checked={measureMethod === 'fast'}
                  onChange={() => setMeasureMethod('fast')}
                  className="text-blue-600"
                />
                <div>
                  <div className="text-xs font-medium text-gray-900">Fast Ellipse</div>
                  <div className="text-[10px] text-gray-500">~178 org/sec, recommended</div>
                </div>
              </label>
              <label className="flex items-center gap-2 p-2 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="radio"
                  value="sam"
                  checked={measureMethod === 'sam'}
                  onChange={() => setMeasureMethod('sam')}
                  className="text-blue-600"
                />
                <div>
                  <div className="text-xs font-medium text-gray-900">SAM Contours</div>
                  <div className="text-[10px] text-gray-500">~1 org/sec, more accurate</div>
                </div>
              </label>
            </div>

            <button
              onClick={onRunMeasurement}
              disabled={measureJob?.status === 'running'}
              className={`w-full px-4 py-2 rounded-lg font-medium transition-colors text-sm disabled:bg-gray-300 disabled:cursor-not-allowed ${
                measurementDone
                  ? 'border border-gray-300 text-gray-700 hover:bg-gray-50'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {measureJob?.status === 'running'
                ? 'Running…'
                : measurementDone
                ? 'Re-run Measurement'
                : 'Run Measurement'}
            </button>

            {measureJob && !measurementDone && <JobProgress job={measureJob} />}

            {measurementDone && (
              <div className="space-y-2">
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-start gap-2">
                  <CheckCircle2 size={15} className="text-green-600 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs font-medium text-green-800">Measurement complete</p>
                    {csvDataLength > 0 && (
                      <p className="text-xs text-green-700 mt-0.5">{csvDataLength} organisms measured</p>
                    )}
                  </div>
                </div>
                <button
                  onClick={onViewResults}
                  className="w-full px-4 py-2 border border-blue-300 text-blue-700 rounded-lg hover:bg-blue-50 font-medium transition-colors text-sm flex items-center justify-center gap-2"
                >
                  <Table2 size={14} />
                  View Results
                </button>
              </div>
            )}

            {measureError && <p className="text-xs text-red-600">{measureError}</p>}
          </div>
        )}
      </div>

      {/* Reset at bottom */}
      <div className="shrink-0 px-4 pb-4 pt-2 border-t">
        <button
          onClick={onReset}
          className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50 text-sm flex items-center justify-center gap-2 transition-colors"
        >
          <RefreshCw size={13} />
          Reset Workspace
        </button>
      </div>
    </div>
  )
}
