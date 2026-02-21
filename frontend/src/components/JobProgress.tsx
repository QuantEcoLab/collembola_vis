import { Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react'
import type { Job } from '../api/types'

interface Props {
  job: Job | undefined
}

export default function JobProgress({ job }: Props) {
  if (!job) return null

  const pct = Math.round(job.progress * 100)
  const isActive = job.status === 'running' || job.status === 'pending'
  // Show indeterminate animation when there's no measurable progress yet
  const isIndeterminate = isActive && pct === 0

  return (
    <div className="bg-white border rounded-lg p-4 space-y-3">
      {/* Status row */}
      <div className="flex items-center gap-2 min-w-0">
        {job.status === 'running' && (
          <Loader2 size={16} className="animate-spin text-blue-500 shrink-0" />
        )}
        {job.status === 'completed' && (
          <CheckCircle2 size={16} className="text-green-600 shrink-0" />
        )}
        {job.status === 'failed' && (
          <XCircle size={16} className="text-red-600 shrink-0" />
        )}
        {job.status === 'pending' && (
          <Clock size={16} className="text-gray-400 shrink-0" />
        )}

        <span className="text-sm font-medium capitalize text-gray-700">
          {job.status}
        </span>

        {job.message && (
          <span className="text-sm text-gray-500 truncate">{job.message}</span>
        )}

        {isActive && !isIndeterminate && (
          <span className="ml-auto text-sm font-mono font-medium text-blue-600 shrink-0">
            {pct}%
          </span>
        )}
      </div>

      {/* Progress bar */}
      {isActive && (
        <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
          {isIndeterminate ? (
            <div className="h-3 w-full bg-gradient-to-r from-blue-300 via-blue-500 to-blue-300 animate-pulse rounded-full" />
          ) : (
            <div
              className="bg-blue-500 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${pct}%` }}
            />
          )}
        </div>
      )}

      {job.error && (
        <p className="text-sm text-red-600">{job.error}</p>
      )}
    </div>
  )
}
