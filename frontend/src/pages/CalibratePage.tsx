import { useState } from 'react'
import ImageUploader from '../components/ImageUploader'
import ImageViewer from '../components/ImageViewer'
import CalibrationTool from '../components/CalibrationTool'
import { autoCalibrate, manualCalibrate, thumbnailUrl } from '../api/client'
import { useCalibrationStore } from '../store/calibrationStore'
import type { ImageInfo, CalibrationResult } from '../api/types'
import { Ruler, Wand2 } from 'lucide-react'

export default function CalibratePage() {
  const [image, setImage] = useState<ImageInfo | null>(null)
  const [result, setResult] = useState<CalibrationResult | null>(null)
  const [manualMode, setManualMode] = useState(false)
  const [rulerMm, setRulerMm] = useState(10)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const setCalibration = useCalibrationStore((s) => s.setCalibration)
  const storedUm = useCalibrationStore((s) => s.umPerPixel)

  const handleAutoCalibrate = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    try {
      const res = await autoCalibrate(image.image_id, rulerMm)
      setResult(res)
      if (res.um_per_pixel) {
        setCalibration(res.um_per_pixel, res.calibration_id!, res.method, res.confidence)
      }
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleManualPoints = async (p1: [number, number], p2: [number, number]) => {
    if (!image) return
    setLoading(true)
    setError(null)
    try {
      const res = await manualCalibrate(image.image_id, p1, p2, rulerMm)
      setResult(res)
      if (res.um_per_pixel) {
        setCalibration(res.um_per_pixel, res.calibration_id!, res.method, res.confidence)
      }
      setManualMode(false)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const scale = image ? image.width / (image.thumbnail_width || image.width) : 1

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Calibration</h2>
        <p className="text-sm text-gray-500 mt-1">
          Set the micrometer-per-pixel scale using a ruler in your image
        </p>
      </div>

      {!image && <ImageUploader onUploaded={setImage} />}

      {image && (
        <>
          <div className="relative h-[400px]">
            <ImageViewer
              src={thumbnailUrl(image.image_id)}
              className="h-full"
              originalWidth={image.width}
              originalHeight={image.height}
            />
            {manualMode && (
              <CalibrationTool
                scale={scale}
                onPointsSelected={handleManualPoints}
              />
            )}
          </div>

          <div className="flex items-end gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Ruler length (mm)
              </label>
              <input
                type="number"
                value={rulerMm}
                onChange={(e) => setRulerMm(Number(e.target.value))}
                className="w-28 border rounded-lg px-3 py-2 text-sm"
                min={0.1}
                step={0.1}
              />
            </div>

            <button
              onClick={handleAutoCalibrate}
              disabled={loading}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium"
            >
              <Wand2 size={16} />
              Auto Calibrate
            </button>

            <button
              onClick={() => setManualMode(!manualMode)}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium border ${
                manualMode
                  ? 'bg-orange-50 border-orange-300 text-orange-700'
                  : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Ruler size={16} />
              {manualMode ? 'Click two ruler points...' : 'Manual Calibrate'}
            </button>

            <button
              onClick={() => {
                setImage(null)
                setResult(null)
                setManualMode(false)
              }}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
            >
              Change image
            </button>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          {result && (
            <div className={`rounded-lg p-4 border ${
              result.um_per_pixel ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'
            }`}>
              {result.um_per_pixel ? (
                <>
                  <p className="font-medium text-green-800">
                    Calibration: {result.um_per_pixel} um/pixel
                  </p>
                  <p className="text-sm text-green-600 mt-1">
                    Method: {result.method} | Confidence: {(result.confidence * 100).toFixed(0)}%
                    {result.ruler_px && ` | Ruler: ${result.ruler_px}px`}
                  </p>
                </>
              ) : (
                <p className="text-yellow-800">
                  Auto calibration failed: {result.error}. Try manual calibration.
                </p>
              )}
            </div>
          )}

          {storedUm && !result && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-sm text-blue-800">
                Active calibration: <strong>{storedUm} um/pixel</strong>
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}
