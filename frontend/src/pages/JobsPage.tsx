import { useEffect, useState } from 'react'
import { listJobs } from '../api/client'
import type { Job } from '../api/types'
import { Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react'

const statusIcon: Record<string, React.ReactNode> = {
  pending: <Clock size={16} className="text-gray-400" />,
  running: <Loader2 size={16} className="animate-spin text-blue-500" />,
  completed: <CheckCircle2 size={16} className="text-green-500" />,
  failed: <XCircle size={16} className="text-red-500" />,
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([])

  useEffect(() => {
    const load = () => listJobs().then(setJobs).catch(() => {})
    load()
    const iv = setInterval(load, 3000)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Jobs</h2>
        <p className="text-sm text-gray-500 mt-1">All submitted jobs</p>
      </div>

      {jobs.length === 0 ? (
        <p className="text-sm text-gray-500">No jobs yet.</p>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Status</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">ID</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Type</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Progress</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Message</th>
                <th className="px-4 py-2 text-left font-medium text-gray-600">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {jobs.map((job) => (
                <tr key={job.id} className="hover:bg-gray-50">
                  <td className="px-4 py-2">{statusIcon[job.status]}</td>
                  <td className="px-4 py-2 font-mono text-xs">{job.id}</td>
                  <td className="px-4 py-2 capitalize">{job.type}</td>
                  <td className="px-4 py-2">{Math.round(job.progress * 100)}%</td>
                  <td className="px-4 py-2 text-gray-500 truncate max-w-[200px]">
                    {job.error || job.message}
                  </td>
                  <td className="px-4 py-2 text-gray-500">
                    {new Date(job.created_at).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
