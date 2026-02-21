import { useState, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import JobProgress from '../components/JobProgress'
import MeasurementTable from '../components/MeasurementTable'
import ExportButton from '../components/ExportButton'
import { runMeasurement, outputFileUrl } from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import { useCalibrationStore } from '../store/calibrationStore'
import { Crosshair } from 'lucide-react'

export default function MeasurePage() {
  const location = useLocation()
  const state = location.state as {
    imageId?: string
    imagePath?: string
    detectionsCSV?: string
  } | null

  const storedUm = useCalibrationStore((s) => s.umPerPixel)

  const [imageId, setImageId] = useState(state?.imageId || '')
  const [detectionsCSV, setDetectionsCSV] = useState(state?.detectionsCSV || '')
  const [umPerPixel, setUmPerPixel] = useState(storedUm || 8.57)
  const [method, setMethod] = useState<'fast' | 'sam'>('fast')
  const [jobId, setJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [csvData, setCsvData] = useState<Record<string, any>[] | null>(null)
  const job = useJobProgress(jobId)

  // When the job completes, fetch the CSV data
  useEffect(() => {
    if (job?.status === 'completed' && job.result?.csv_path) {
      const csvUrl = outputFileUrl(job.id, `${job.result.csv_path.split('/').pop()}`)
      fetch(csvUrl)
        .then((r) => r.text())
        .then((text) => {
          const lines = text.trim().split('\n')
          const headers = lines[0].split(',')
          const rows = lines.slice(1).map((line) => {
            const vals = line.split(',')
            const row: Record<string, any> = {}
            headers.forEach((h, i) => {
              const v = vals[i]
              const num = Number(v)
              row[h] = isNaN(num) ? v : num
            })
            return row
          })
          setCsvData(rows)
        })
        .catch(() => {})
    }
  }, [job?.status, job?.result?.csv_path, job?.id])

  const handleRun = async () => {
    if (!imageId || !detectionsCSV) return
    setError(null)
    setCsvData(null)
    try {
      const res = await runMeasurement({
        image_id: imageId,
        detections_csv: detectionsCSV,
        um_per_pixel: umPerPixel,
        method,
      })
      setJobId(res.job_id)
    } catch (e: any) {
      setError(e.message)
    }
  }

  return (
    <div className="max-w-5xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Measurement</h2>
        <p className="text-sm text-gray-500 mt-1">
          Measure organism morphology from detection results
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Image ID</label>
          <input
            type="text"
            value={imageId}
            onChange={(e) => setImageId(e.target.value)}
            className="w-full border rounded-lg px-3 py-2 text-sm"
            placeholder="From detection step"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Detections CSV</label>
          <input
            type="text"
            value={detectionsCSV}
            onChange={(e) => setDetectionsCSV(e.target.value)}
            className="w-full border rounded-lg px-3 py-2 text-sm"
            placeholder="Path to detections CSV"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            um/pixel {storedUm && <span className="text-blue-600">(calibrated)</span>}
          </label>
          <input
            type="number"
            value={umPerPixel}
            onChange={(e) => setUmPerPixel(Number(e.target.value))}
            className="w-full border rounded-lg px-3 py-2 text-sm"
            min={0.01}
            step={0.01}
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex gap-2">
          <button
            onClick={() => setMethod('fast')}
            className={`px-3 py-1.5 rounded-lg text-sm border ${
              method === 'fast'
                ? 'bg-blue-50 border-blue-300 text-blue-700'
                : 'bg-white border-gray-300 text-gray-600'
            }`}
          >
            Fast (ellipse)
          </button>
          <button
            onClick={() => setMethod('sam')}
            className={`px-3 py-1.5 rounded-lg text-sm border ${
              method === 'sam'
                ? 'bg-blue-50 border-blue-300 text-blue-700'
                : 'bg-white border-gray-300 text-gray-600'
            }`}
          >
            Accurate (SAM)
          </button>
        </div>

        <button
          onClick={handleRun}
          disabled={!imageId || !detectionsCSV || (!!jobId && job?.status === 'running')}
          className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium"
        >
          <Crosshair size={16} />
          Run Measurements
        </button>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <JobProgress job={job} />

      {job?.status === 'completed' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="font-medium text-green-800">
              Measured {job.result.num_organisms} organisms ({job.result.method} method)
            </p>
            <ExportButton
              url={outputFileUrl(job.id, job.result.csv_path.split('/').pop()!)}
              filename={job.result.csv_path.split('/').pop()}
            />
          </div>

          {csvData && <MeasurementTable data={csvData} />}
        </div>
      )}
    </div>
  )
}
