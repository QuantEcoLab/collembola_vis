import { useEffect, useState } from 'react'
import { listJobs } from '../api/client'
import type { Job } from '../api/types'
import { FileSpreadsheet, ScanSearch, Crosshair } from 'lucide-react'

export default function ResultsPage() {
  const [jobs, setJobs] = useState<Job[]>([])

  useEffect(() => {
    listJobs().then(setJobs).catch(() => {})
  }, [])

  const completed = jobs.filter((j) => j.status === 'completed')

  const typeIcon = (type: string) => {
    if (type === 'detection') return <ScanSearch size={16} className="text-blue-500" />
    if (type === 'measurement') return <Crosshair size={16} className="text-green-500" />
    return <FileSpreadsheet size={16} className="text-gray-500" />
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Results</h2>
        <p className="text-sm text-gray-500 mt-1">Completed jobs and their outputs</p>
      </div>

      {completed.length === 0 ? (
        <p className="text-sm text-gray-500">No completed jobs yet. Run a detection or measurement first.</p>
      ) : (
        <div className="space-y-2">
          {completed.map((job) => (
            <div
              key={job.id}
              className="bg-white border rounded-lg p-4 flex items-center gap-4"
            >
              {typeIcon(job.type)}
              <div className="flex-1">
                <p className="font-medium text-gray-900 text-sm">
                  {job.type === 'detection'
                    ? `Detection: ${job.result.num_detections} organisms`
                    : job.type === 'measurement'
                      ? `Measurement: ${job.result.num_organisms} organisms (${job.result.method})`
                      : job.type}
                </p>
                <p className="text-xs text-gray-500">
                  Job {job.id} | Completed{' '}
                  {job.completed_at && new Date(job.completed_at).toLocaleString()}
                </p>
              </div>
              {job.result.csv_path && (
                <a
                  href={`/files/outputs/${job.id}/${job.result.csv_path.split('/').pop()}`}
                  download
                  className="text-sm text-blue-600 hover:underline"
                >
                  Download CSV
                </a>
              )}
              {job.result.overlay_path && (
                <a
                  href={`/files/outputs/${job.id}/${job.result.overlay_path.split('/').pop()}`}
                  target="_blank"
                  rel="noreferrer"
                  className="text-sm text-blue-600 hover:underline"
                >
                  View overlay
                </a>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
