import { act, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProjectsPage from './ProjectsPage'
import { listProjects } from '../api/client'

const project = { id: 'preserved-project', name: 'Preserved project', description: 'Existing data',
  created_by: 'user1', created_at: '2026-11-04', updated_at: '2026-11-04', image_count: 2, thumbnail_image_id: null }
const response = () => new Response(JSON.stringify([project]), { headers: { 'Content-Type': 'application/json' } })

beforeEach(() => { localStorage.clear() })
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('Project list bounded loading and failures', () => {
  it('ends Loading after ten seconds for a pending fetch, then successfully retries without deleting data', async () => {
    vi.useFakeTimers()
    const fetcher = vi.fn().mockImplementation(() => new Promise(() => {}))
    vi.stubGlobal('fetch', fetcher)
    render(<MemoryRouter><ProjectsPage /></MemoryRouter>)
    expect(screen.getByRole('status')).toHaveTextContent('Loading')
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('/api/projects nije dovršen unutar 10 s')
    expect(screen.queryByText('No projects yet.')).not.toBeInTheDocument()
    expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true)
    expect(fetcher.mock.calls[0][1].cache).toBe('no-store')
    fetcher.mockResolvedValueOnce(response())
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Pokušaj ponovno' })) })
    expect(screen.getByText('Preserved project')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it('also times out when HTTP 200 headers arrive but the JSON body remains pending', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200,
      headers: new Headers({ 'content-type': 'application/json' }), json: () => new Promise(() => {}) }))
    const pending = listProjects()
    const assertion = expect(pending).rejects.toThrow('unutar 10 s')
    await vi.advanceTimersByTimeAsync(10_000)
    await assertion
  })

  it('reports an HTTP error with endpoint and status instead of silently showing an empty database', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'Baza projekata je zaključana' }),
      { status: 503, headers: { 'Content-Type': 'application/json' } })))
    await expect(listProjects()).rejects.toThrow('HTTP 503, /api/projects')
  })

  it('identifies a misrouted HTTP 200 HTML response instead of treating it as project data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>SPA fallback</html>',
      { status: 200, headers: { 'Content-Type': 'text/html' } })))
    await expect(listProjects()).rejects.toThrow('/api/projects vratio je HTTP 200, ali ne JSON')
  })

  it('cancels a pending project request on navigation', async () => {
    const aborter = new AbortController()
    vi.stubGlobal('fetch', vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
    })))
    const pending = listProjects(aborter.signal)
    aborter.abort()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  })
})
