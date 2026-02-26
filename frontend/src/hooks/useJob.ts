import { useEffect, useRef } from 'react'
import { useJobStore } from '../store/jobStore'
import { getJob } from '../api/client'
import type { Job } from '../api/types'

/**
 * Subscribe to real-time job progress via WebSocket with HTTP polling fallback.
 * Returns the latest job state from the store.
 */
export function useJobProgress(jobId: string | null): Job | undefined {
  const updateJob = useJobStore((s) => s.updateJob)
  const job = useJobStore((s) => (jobId ? s.jobs[jobId] : undefined))
  const wsRef = useRef<WebSocket | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const wsConnected = useRef(false)

  useEffect(() => {
    if (!jobId) return

    const stopPoll = () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }

    const startPoll = () => {
      if (pollRef.current) return
      pollRef.current = setInterval(async () => {
        try {
          const data = await getJob(jobId)
          updateJob(data)
          if (data.status === 'completed' || data.status === 'failed') stopPoll()
        } catch {
          // ignore transient errors
        }
      }, 2000)
    }

    // Try WebSocket first
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const raw = localStorage.getItem('auth')
    const token = raw ? ((JSON.parse(raw) as { state?: { token?: string } }).state?.token ?? '') : ''
    const base = import.meta.env.BASE_URL.replace(/\/$/, '')
    const ws = new WebSocket(`${protocol}//${window.location.host}${base}/ws/jobs/${jobId}?token=${encodeURIComponent(token)}`)
    wsRef.current = ws
    wsConnected.current = false

    ws.onopen = () => { wsConnected.current = true }

    ws.onmessage = (event) => {
      const data: Job = JSON.parse(event.data)
      updateJob(data)
      if (data.status === 'completed' || data.status === 'failed') {
        ws.close()
        stopPoll()
      }
    }

    // Fall back to polling if WS fails or doesn't connect within 3s
    const wsTimeout = setTimeout(() => {
      if (!wsConnected.current) startPoll()
    }, 3000)

    ws.onerror = () => {
      ws.close()
      startPoll()
    }

    ws.onclose = () => {
      // Always fall back to polling on close — handles both initial failure
      // and mid-run disconnection (e.g. long SAM jobs that outlast the WS timeout).
      startPoll()
    }

    return () => {
      clearTimeout(wsTimeout)
      ws.close()
      wsRef.current = null
      stopPoll()
    }
  }, [jobId, updateJob])

  return job
}
