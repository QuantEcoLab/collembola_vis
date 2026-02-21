import { useEffect, useRef } from 'react'
import { useJobStore } from '../store/jobStore'
import type { Job } from '../api/types'

/**
 * Subscribe to real-time job progress via WebSocket.
 * Returns the latest job state from the store.
 */
export function useJobProgress(jobId: string | null): Job | undefined {
  const updateJob = useJobStore((s) => s.updateJob)
  const job = useJobStore((s) => (jobId ? s.jobs[jobId] : undefined))
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!jobId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/jobs/${jobId}`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data: Job = JSON.parse(event.data)
      updateJob(data)
      if (data.status === 'completed' || data.status === 'failed') {
        ws.close()
      }
    }

    ws.onerror = () => ws.close()

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [jobId, updateJob])

  return job
}
