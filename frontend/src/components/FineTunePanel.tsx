import { useState, useEffect } from 'react'
import { listModels, runFinetune } from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import JobProgress from './JobProgress'
import type { ModelInfo } from '../api/types'

interface Props {
  imageId: string
  onModelSelected?: (modelPath: string) => void
}

export default function FineTunePanel({
  imageId,
  onModelSelected,
}: Props) {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [baseModel, setBaseModel] = useState<string>('')
  const [epochs, setEpochs] = useState(15)
  const [device, setDevice] = useState('0')
  const [ftJobId, setFtJobId] = useState<string | null>(null)
  const [ftError, setFtError] = useState<string | null>(null)
  const ftJob = useJobProgress(ftJobId)
  const ftDone = ftJob?.status === 'completed'

  useEffect(() => {
    listModels()
      .then((ms) => {
        setModels(ms)
        if (ms.length > 0 && !baseModel) setBaseModel(ms[0].path)
      })
      .catch(() => {})
  }, [])

  // Reload models after fine-tune completes and pre-select new model
  useEffect(() => {
    if (!ftDone) return
    listModels()
      .then((ms) => {
        setModels(ms)
        if (ftJob?.result?.model_path) {
          setBaseModel(ftJob.result.model_path as string)
        } else if (ms.length > 0) {
          setBaseModel(ms[0].path)
        }
      })
      .catch(() => {})
  }, [ftDone])

  const handleRun = async () => {
    setFtError(null)
    try {
      const res = await runFinetune({
        image_id: imageId,
        annotation_file: imageId,
        base_model: baseModel || undefined,
        epochs,
        device,
      })
      setFtJobId(res.job_id)
    } catch (e: any) {
      setFtError(e.message)
    }
  }

  const isRunning = ftJob && (ftJob.status === 'pending' || ftJob.status === 'running')

  return (
    <div className="space-y-3">
      {/* Base model selector */}
      <div className="space-y-1">
        <label className="text-xs text-gray-500">Base model</label>
        <select
          value={baseModel}
          onChange={(e) => setBaseModel(e.target.value)}
          className="w-full border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
          disabled={!!isRunning}
        >
          {models.map((m) => (
            <option key={m.path} value={m.path}>
              {m.name} ({m.size_mb} MB)
            </option>
          ))}
          {models.length === 0 && (
            <option value="">No models found</option>
          )}
        </select>
      </div>

      {/* Epochs */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 w-14 shrink-0">Epochs</span>
        <input
          type="number"
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
          className="w-20 border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
          min={5}
          max={50}
          step={5}
          disabled={!!isRunning}
        />
      </div>

      {/* Device */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 w-14 shrink-0">Device</span>
        <input
          type="text"
          value={device}
          onChange={(e) => setDevice(e.target.value)}
          className="w-20 border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
          placeholder="0 / cpu"
          disabled={!!isRunning}
        />
      </div>

      {/* Run button / progress */}
      {!ftJob || ftJob.status === 'failed' ? (
        <>
          <button
            onClick={handleRun}
            disabled={!baseModel}
            className="w-full py-2 bg-violet-600 text-white rounded-lg text-sm font-medium hover:bg-violet-700 disabled:opacity-50 transition-colors"
          >
            Fine-tune Model
          </button>
          {ftError && <p className="text-xs text-red-600">{ftError}</p>}
          {ftJob?.status === 'failed' && ftJob.error && (
            <p className="text-xs text-red-600">{ftJob.error}</p>
          )}
        </>
      ) : (
        <JobProgress job={ftJob} />
      )}

      {/* Completion */}
      {ftDone && ftJob.result && (
        <div className="rounded-lg bg-violet-50 border border-violet-200 p-2.5 text-xs text-violet-800 space-y-1">
          <p className="font-medium">✓ Fine-tune complete</p>
          <p className="truncate">{ftJob.result.model_name as string}</p>
          {ftJob.result.map50 != null && (
            <p>mAP50: {((ftJob.result.map50 as number) * 100).toFixed(1)}%</p>
          )}
          {onModelSelected && (
            <button
              onClick={() => onModelSelected(ftJob.result.model_path as string)}
              className="mt-1 text-violet-700 underline hover:text-violet-900"
            >
              Use this model
            </button>
          )}
        </div>
      )}
    </div>
  )
}
