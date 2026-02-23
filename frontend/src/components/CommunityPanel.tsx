import { useState, useEffect, useCallback } from 'react'
import { Download, ChevronLeft, ChevronRight } from 'lucide-react'
import {
  listCommunity,
  getCommunityStats,
  submitToCommunity,
  communityExportUrl,
  type CommunitySubmitRequest,
} from '../api/client'
import type { CommunityEntry, CommunityStats } from '../api/types'

interface Props {
  /** Boxes to submit (already filtered by caller to non-rejected) */
  submitData: CommunitySubmitRequest | null
}

const PAGE_SIZE = 20

export default function CommunityPanel({ submitData }: Props) {
  const [stats, setStats] = useState<CommunityStats | null>(null)
  const [entries, setEntries] = useState<CommunityEntry[]>([])
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'submitting' | 'ok' | 'error'>('idle')
  const [submitError, setSubmitError] = useState<string | null>(null)

  const load = useCallback(async (off: number) => {
    setLoading(true)
    try {
      const [s, e] = await Promise.all([
        getCommunityStats(),
        listCommunity(PAGE_SIZE, off),
      ])
      setStats(s)
      setEntries(e)
      setHasMore(e.length === PAGE_SIZE)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load(0) }, [load])

  const handleSubmit = async () => {
    if (!submitData) return
    setSubmitStatus('submitting')
    setSubmitError(null)
    try {
      await submitToCommunity(submitData)
      setSubmitStatus('ok')
      setOffset(0)
      load(0)
    } catch (e: any) {
      setSubmitStatus('error')
      setSubmitError(e.message)
    }
  }

  const goPage = (off: number) => {
    setOffset(off)
    load(off)
  }

  const fmt = (val: number | null | undefined, dec = 2) =>
    val == null ? '—' : val.toFixed(dec)

  const fmtDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString()
    } catch {
      return iso
    }
  }

  return (
    <div className="space-y-3">
      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-2 gap-2">
          {([
            ['Submissions', stats.total_submissions],
            ['Organisms', stats.total_detections],
            ['Users', stats.total_users],
            ['Images', stats.total_images],
          ] as [string, number][]).map(([label, val]) => (
            <div key={label} className="bg-gray-50 rounded-lg p-2 text-center">
              <p className="text-sm font-semibold text-gray-800">{val.toLocaleString()}</p>
              <p className="text-xs text-gray-400">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Submit button */}
      {submitData && (
        <div className="space-y-1">
          <button
            onClick={handleSubmit}
            disabled={submitStatus === 'submitting' || submitStatus === 'ok'}
            className="w-full py-2 bg-teal-600 text-white rounded-lg text-sm font-medium hover:bg-teal-700 disabled:opacity-50 transition-colors"
          >
            {submitStatus === 'submitting'
              ? 'Submitting…'
              : submitStatus === 'ok'
              ? '✓ Submitted!'
              : `Submit ${submitData.boxes.length} detections`}
          </button>
          {submitStatus === 'error' && submitError && (
            <p className="text-xs text-red-600">{submitError}</p>
          )}
        </div>
      )}

      {/* Table */}
      {loading && entries.length === 0 ? (
        <p className="text-xs text-gray-400">Loading…</p>
      ) : entries.length === 0 ? (
        <p className="text-xs text-gray-400">No submissions yet.</p>
      ) : (
        <div className="space-y-1">
          {entries.map((e) => (
            <div
              key={e.id}
              className="border border-gray-100 rounded-lg p-2 text-xs space-y-0.5 hover:bg-gray-50"
            >
              <div className="flex items-start justify-between gap-1">
                <span className="font-medium text-gray-700 truncate max-w-[60%]" title={e.image_name}>
                  {e.image_name.length > 24 ? `…${e.image_name.slice(-22)}` : e.image_name}
                </span>
                <span className="text-gray-400 shrink-0">{fmtDate(e.submitted_at)}</span>
              </div>
              <div className="flex items-center gap-3 text-gray-500">
                <span>by <span className="font-medium">{e.username}</span></span>
                <span>{e.num_detections} org.</span>
                {e.um_per_pixel != null && <span>{fmt(e.um_per_pixel)} µm/px</span>}
              </div>
              <div className="flex gap-2 pt-0.5">
                <a
                  href={communityExportUrl(e.id, 'json')}
                  download
                  className="flex items-center gap-0.5 text-blue-600 hover:underline"
                >
                  <Download size={10} /> JSON
                </a>
                <a
                  href={communityExportUrl(e.id, 'csv')}
                  download
                  className="flex items-center gap-0.5 text-blue-600 hover:underline"
                >
                  <Download size={10} /> CSV
                </a>
              </div>
            </div>
          ))}

          {/* Pagination */}
          <div className="flex items-center justify-between pt-1">
            <button
              onClick={() => goPage(Math.max(0, offset - PAGE_SIZE))}
              disabled={offset === 0}
              className="flex items-center gap-0.5 text-xs text-gray-400 hover:text-gray-600 disabled:opacity-30"
            >
              <ChevronLeft size={12} /> Prev
            </button>
            <span className="text-xs text-gray-400">
              {offset + 1}–{offset + entries.length}
            </span>
            <button
              onClick={() => goPage(offset + PAGE_SIZE)}
              disabled={!hasMore}
              className="flex items-center gap-0.5 text-xs text-gray-400 hover:text-gray-600 disabled:opacity-30"
            >
              Next <ChevronRight size={12} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
