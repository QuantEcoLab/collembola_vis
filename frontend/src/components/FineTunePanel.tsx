import { useState, useEffect } from 'react'
import { listModels, runFinetune, runFinetuneAll } from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import JobProgress from './JobProgress'
import type { ModelInfo } from '../api/types'

interface Props {
  imageId?: string  // if provided, "This image" tab is available
  onModelSelected?: (modelPath: string) => void
}

type Tab = 'single' | 'all'

export default function FineTunePanel({ imageId, onModelSelected }: Props) {
  const [tab, setTab] = useState<Tab>(imageId ? 'single' : 'all')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [baseModel, setBaseModel] = useState<string>('')
  const [epochs, setEpochs] = useState(15)
  const [epochsAll, setEpochsAll] = useState(20)
  const [device, setDevice] = useState('0')
  const [minAdded, setMinAdded] = useState(10)

  const [ftJobId, setFtJobId] = useState<string | null>(null)
  const [ftAllJobId, setFtAllJobId] = useState<string | null>(null)
  const [ftError, setFtError] = useState<string | null>(null)
  const [ftAllError, setFtAllError] = useState<string | null>(null)

  const ftJob = useJobProgress(ftJobId)
  const ftAllJob = useJobProgress(ftAllJobId)
  const ftDone = ftJob?.status === 'completed'
  const ftAllDone = ftAllJob?.status === 'completed'

  useEffect(() => {
    listModels()
      .then((ms) => {
        setModels(ms)
        if (ms.length > 0) setBaseModel(ms[0].path)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!ftDone && !ftAllDone) return
    listModels()
      .then((ms) => {
        setModels(ms)
        const result = ftDone ? ftJob?.result : ftAllJob?.result
        if (result?.model_path) {
          setBaseModel(result.model_path as string)
        } else if (ms.length > 0) {
          setBaseModel(ms[0].path)
        }
      })
      .catch(() => {})
  }, [ftDone, ftAllDone])

  const handleRunSingle = async () => {
    if (!imageId) return
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

  const handleRunAll = async () => {
    setFtAllError(null)
    try {
      const res = await runFinetuneAll({
        base_model: baseModel || undefined,
        epochs: epochsAll,
        device,
        min_added: minAdded,
      })
      setFtAllJobId(res.job_id)
    } catch (e: any) {
      setFtAllError(e.message)
    }
  }

  const isSingleRunning = ftJob && (ftJob.status === 'pending' || ftJob.status === 'running')
  const isAllRunning = ftAllJob && (ftAllJob.status === 'pending' || ftAllJob.status === 'running')

  return (
    <div className="space-y-3">
      {/* Tab switcher — only show if both tabs are available */}
      {imageId && (
        <div className="flex rounded-lg border overflow-hidden text-xs">
          <button
            onClick={() => setTab('single')}
            className={`flex-1 py-1.5 font-medium transition-colors ${
              tab === 'single' ? 'bg-violet-600 text-white' : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            This image
          </button>
          <button
            onClick={() => setTab('all')}
            className={`flex-1 py-1.5 font-medium transition-colors ${
              tab === 'all' ? 'bg-violet-600 text-white' : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            All annotations
          </button>
        </div>
      )}

      {/* Base model selector */}
      <div className="space-y-1">
        <label className="text-xs text-gray-500">Base model</label>
        <select
          value={baseModel}
          onChange={(e) => setBaseModel(e.target.value)}
          className="w-full border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-violet-400"
          disabled={!!(isSingleRunning || isAllRunning)}
        >
          {models.map((m) => (
            <option key={m.path} value={m.path}>
              {m.name} ({m.size_mb} MB)
            </option>
          ))}
          {models.length === 0 && <option value="">No models found</option>}
        </select>
      </div>

      {/* Device */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 w-14 shrink-0">Device</span>
        <input
          type="text"
          value={device}
          onChange={(e) => setDevice(e.target.value)}
          className="w-20 border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-violet-400"
          placeholder="0 / cpu"
          disabled={!!(isSingleRunning || isAllRunning)}
        />
      </div>

      {/* ── Single image tab ── */}
      {tab === 'single' && imageId && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-14 shrink-0">Epochs</span>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              className="w-20 border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-violet-400"
              min={5} max={50} step={5}
              disabled={!!isSingleRunning}
            />
          </div>

          {!ftJob || ftJob.status === 'failed' ? (
            <>
              <button
                onClick={handleRunSingle}
                disabled={!baseModel || !!isSingleRunning}
                className="w-full py-2 bg-violet-600 text-white rounded-lg text-sm font-medium hover:bg-violet-700 disabled:opacity-50 transition-colors"
              >
                Fine-tune on this image
              </button>
              {ftError && <p className="text-xs text-red-600">{ftError}</p>}
              {ftJob?.status === 'failed' && ftJob.error && (
                <p className="text-xs text-red-600">{ftJob.error}</p>
              )}
            </>
          ) : (
            <JobProgress job={ftJob} />
          )}

          {ftDone && ftJob.result && (
            <CompletionCard job={ftJob} onModelSelected={onModelSelected} />
          )}
        </div>
      )}

      {/* ── All annotations tab ── */}
      {tab === 'all' && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-14 shrink-0">Epochs</span>
            <input
              type="number"
              value={epochsAll}
              onChange={(e) => setEpochsAll(Number(e.target.value))}
              className="w-20 border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-violet-400"
              min={5} max={100} step={5}
              disabled={!!isAllRunning}
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-14 shrink-0">Min added</span>
            <input
              type="number"
              value={minAdded}
              onChange={(e) => setMinAdded(Number(e.target.value))}
              className="w-20 border rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-violet-400"
              min={1} max={200} step={5}
              disabled={!!isAllRunning}
            />
            <span className="text-[10px] text-gray-400">boxes/image</span>
          </div>

          <p className="text-[10px] text-gray-400 leading-tight">
            Only images where users drew more than this many new boxes are included.
            Currently ~20 images qualify at threshold &gt; 10.
          </p>

          {!ftAllJob || ftAllJob.status === 'failed' ? (
            <>
              <button
                onClick={handleRunAll}
                disabled={!baseModel || !!isAllRunning}
                className="w-full py-2 bg-violet-600 text-white rounded-lg text-sm font-medium hover:bg-violet-700 disabled:opacity-50 transition-colors"
              >
                Fine-tune on all annotations
              </button>
              {ftAllError && <p className="text-xs text-red-600">{ftAllError}</p>}
              {ftAllJob?.status === 'failed' && ftAllJob.error && (
                <p className="text-xs text-red-600">{ftAllJob.error}</p>
              )}
            </>
          ) : (
            <JobProgress job={ftAllJob} />
          )}

          {ftAllDone && ftAllJob.result && (
            <CompletionCard job={ftAllJob} onModelSelected={onModelSelected} />
          )}
        </div>
      )}
    </div>
  )
}

function CompletionCard({
  job,
  onModelSelected,
}: {
  job: { result: Record<string, unknown> }
  onModelSelected?: (path: string) => void
}) {
  const r = job.result
  return (
    <div className="rounded-lg bg-violet-50 border border-violet-200 p-2.5 text-xs text-violet-800 space-y-1">
      <p className="font-medium">Fine-tune complete</p>
      <p className="truncate">{r.model_name as string}</p>
      {r.map50 != null && (
        <p>mAP50: {((r.map50 as number) * 100).toFixed(1)}%</p>
      )}
      {r.images_used != null && (
        <p>{r.images_used as number} images · {r.tiles_used as number} tiles</p>
      )}
      {onModelSelected && (
        <button
          onClick={() => onModelSelected(r.model_path as string)}
          className="mt-1 text-violet-700 underline hover:text-violet-900"
        >
          Use this model
        </button>
      )}
    </div>
  )
}
