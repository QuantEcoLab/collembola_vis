import { Loader2, CheckCircle2, XCircle } from 'lucide-react'
import type { Job } from '../api/types'

interface Props {
  job: Job | undefined
}

export default function JobProgress({ job }: Props) {
  if (!job) return null

  const pct = Math.round(job.progress * 100)

  return (
    <div className="bg-white border rounded-lg p-4 space-y-2">
      <div className="flex items-center gap-2">
        {job.status === 'running' && <Loader2 size={18} className="animate-spin text-blue-500" />}
        {job.status === 'completed' && <CheckCircle2 size={18} className="text-green-600" />}
        {job.status === 'failed' && <XCircle size={18} className="text-red-600" />}
        {job.status === 'pending' && <Loader2 size={18} className="text-gray-400" />}
        <span className="text-sm font-medium capitalize">{job.status}</span>
        {job.message && (
          <span className="text-sm text-gray-500 ml-2">{job.message}</span>
        )}
      </div>

      {(job.status === 'running' || job.status === 'pending') && (
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}

      {job.error && (
        <p className="text-sm text-red-600">{job.error}</p>
      )}
    </div>
  )
}
