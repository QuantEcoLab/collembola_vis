import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogOut, Download, ChevronLeft, ChevronRight, RefreshCw, Search, FolderOpen } from 'lucide-react'
import { listCommunity, getCommunityStats, communityExportUrl, getCommunityEntry } from '../api/client'
import { useAuthStore } from '../store/authStore'
import { useCommunityLoadStore } from '../store/communityLoadStore'
import type { CommunityEntry, CommunityStats, AnnotatedBox } from '../api/types'

const PAGE_SIZE = 20

export default function CommunityPage() {
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const setPending = useCommunityLoadStore((s) => s.setPending)

  const [stats, setStats] = useState<CommunityStats | null>(null)
  const [entries, setEntries] = useState<CommunityEntry[]>([])
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [search, setSearch] = useState('')
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const load = useCallback(async (off: number, q: string) => {
    setLoading(true)
    try {
      const [s, e] = await Promise.all([
        getCommunityStats(),
        listCommunity(PAGE_SIZE, off, q),
      ])
      setStats(s)
      setEntries(e)
      setHasMore(e.length === PAGE_SIZE)
    } catch {
      // auth errors redirect automatically
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load(0, '') }, [load])

  const handleSearch = (val: string) => {
    setSearch(val)
    setOffset(0)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => load(0, val), 300)
  }

  const goPage = (off: number) => {
    setOffset(off)
    load(off, search)
  }

  const handleLoad = async (entry: CommunityEntry) => {
    setLoadingId(entry.id)
    try {
      const full = await getCommunityEntry(entry.id)
      // Map raw boxes to AnnotatedBox, defaulting missing fields
      const boxes: AnnotatedBox[] = full.boxes.map((b: any, i: number) => ({
        id: b.id ?? `community-${i}`,
        x1: b.x1,
        y1: b.y1,
        x2: b.x2,
        y2: b.y2,
        conf: b.conf ?? 1.0,
        status: 'accepted' as const,
      }))
      setPending(boxes, entry.image_name, entry.username)
      navigate('/')
    } catch {
      // ignore
    } finally {
      setLoadingId(null)
    }
  }

  const fmtDate = (iso: string) => {
    try { return new Date(iso).toLocaleDateString() } catch { return iso }
  }

  const fmtNum = (v: number | null) => v == null ? '—' : v.toFixed(2)

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      {/* Header */}
      <header className="shrink-0 bg-white border-b px-5 h-12 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <span className="font-semibold text-gray-900">Collembola</span>
          <nav className="flex items-center gap-1">
            <Link
              to="/"
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Workspace
            </Link>
            <Link
              to="/community"
              className="text-sm px-3 py-1 rounded-md bg-gray-100 text-gray-900 font-medium transition-colors"
            >
              Community
            </Link>
          </nav>
        </div>
        <button
          onClick={() => { logout(); navigate('/login', { replace: true }) }}
          className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700"
        >
          <LogOut size={14} />
          Sign out
        </button>
      </header>

      {/* Body */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">

          {/* Page title */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Community Database</h1>
              <p className="text-sm text-gray-500 mt-0.5">
                Curated detection results submitted by all users
              </p>
            </div>
            <button
              onClick={() => load(offset, search)}
              disabled={loading}
              className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700 disabled:opacity-50"
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              Refresh
            </button>
          </div>

          {/* Stats bar */}
          {stats && (
            <div className="grid grid-cols-4 gap-4">
              {([
                ['Submissions', stats.total_submissions],
                ['Organisms', stats.total_detections],
                ['Contributors', stats.total_users],
                ['Images', stats.total_images],
              ] as [string, number][]).map(([label, val]) => (
                <div key={label} className="bg-white border rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-gray-800">{val.toLocaleString()}</p>
                  <p className="text-sm text-gray-400 mt-0.5">{label}</p>
                </div>
              ))}
            </div>
          )}

          {/* Search */}
          <div className="relative">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            <input
              type="text"
              value={search}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="Search by image name or user…"
              className="w-full pl-9 pr-4 py-2 border border-gray-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </div>

          {/* Table */}
          <div className="bg-white border rounded-xl overflow-hidden">
            {loading && entries.length === 0 ? (
              <div className="p-12 text-center text-sm text-gray-400">Loading…</div>
            ) : entries.length === 0 ? (
              <div className="p-12 text-center">
                {search ? (
                  <p className="text-sm text-gray-400">No results for "{search}".</p>
                ) : (
                  <>
                    <p className="text-sm text-gray-400">No submissions yet.</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Run a detection in the{' '}
                      <Link to="/" className="text-blue-500 hover:underline">Workspace</Link>
                      {' '}and submit your results.
                    </p>
                  </>
                )}
              </div>
            ) : (
              <>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-gray-50 text-xs text-gray-400 uppercase tracking-wide">
                      <th className="text-left px-5 py-3 font-medium">Date</th>
                      <th className="text-left px-5 py-3 font-medium">User</th>
                      <th className="text-left px-5 py-3 font-medium">Image</th>
                      <th className="text-right px-5 py-3 font-medium">Organisms</th>
                      <th className="text-right px-5 py-3 font-medium">µm/px</th>
                      <th className="text-center px-5 py-3 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {entries.map((e) => (
                      <tr key={e.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-5 py-3 text-gray-500 whitespace-nowrap">
                          {fmtDate(e.submitted_at)}
                        </td>
                        <td className="px-5 py-3 font-medium text-gray-700">
                          {e.username}
                        </td>
                        <td className="px-5 py-3 text-gray-600 max-w-xs">
                          <span className="truncate block" title={e.image_name}>
                            {e.image_name}
                          </span>
                        </td>
                        <td className="px-5 py-3 text-right font-medium text-gray-800">
                          {e.num_detections.toLocaleString()}
                        </td>
                        <td className="px-5 py-3 text-right text-gray-500">
                          {fmtNum(e.um_per_pixel)}
                        </td>
                        <td className="px-5 py-3">
                          <div className="flex items-center justify-center gap-3">
                            <button
                              onClick={() => handleLoad(e)}
                              disabled={loadingId === e.id}
                              className="flex items-center gap-1 text-teal-600 hover:text-teal-800 text-xs font-medium disabled:opacity-50"
                              title="Load boxes into workspace"
                            >
                              <FolderOpen size={12} />
                              {loadingId === e.id ? '…' : 'Load'}
                            </button>
                            <a
                              href={communityExportUrl(e.id, 'json')}
                              download
                              className="flex items-center gap-1 text-blue-600 hover:text-blue-800 text-xs font-medium"
                            >
                              <Download size={12} />
                              JSON
                            </a>
                            <a
                              href={communityExportUrl(e.id, 'csv')}
                              download
                              className="flex items-center gap-1 text-blue-600 hover:text-blue-800 text-xs font-medium"
                            >
                              <Download size={12} />
                              CSV
                            </a>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* Pagination */}
                <div className="flex items-center justify-between px-5 py-3 border-t bg-gray-50">
                  <button
                    onClick={() => goPage(Math.max(0, offset - PAGE_SIZE))}
                    disabled={offset === 0}
                    className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-800 disabled:opacity-30"
                  >
                    <ChevronLeft size={16} /> Previous
                  </button>
                  <span className="text-xs text-gray-400">
                    {offset + 1}–{offset + entries.length}
                    {search && <span className="ml-1 text-gray-300">· filtered</span>}
                  </span>
                  <button
                    onClick={() => goPage(offset + PAGE_SIZE)}
                    disabled={!hasMore}
                    className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-800 disabled:opacity-30"
                  >
                    Next <ChevronRight size={16} />
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
