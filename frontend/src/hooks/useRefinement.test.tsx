import { render, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useRefinement } from './useRefinement'
import type { AnnotatedBox } from '../api/types'
import * as api from '../api/client'

vi.mock('../api/client', () => ({ getAnnotations: vi.fn(), getDetectionBoxes: vi.fn(), saveAnnotations: vi.fn() }))

const box = (id: string, x1: number): AnnotatedBox => ({ id, x1, y1: 10, x2: x1 + 50, y2: 60, conf: .9, status: 'accepted' })
const annotation = (imageId: string, boxes: AnnotatedBox[]) => ({
  image_id: imageId, image_filename: `${imageId}.jpg`, source_job_id: `det-${imageId}`, created_at: '', boxes })

type Render = { imageId: string | null; ids: string[]; saved: boolean; source: string | null; error: string | null; loading: boolean }

function Probe({ imageId, jobId, done, log }: { imageId: string | null; jobId: string | null; done: boolean; log: Render[] }) {
  const refinement = useRefinement(imageId, jobId, done)
  log.push({ imageId, ids: refinement.boxes.map(b => b.id), saved: refinement.annotationsSaved,
    source: refinement.restoredSourceJobId, error: refinement.error, loading: refinement.isLoading })
  return null
}

const notFound = () => Promise.reject(new Error('No annotations found for this image'))

beforeEach(() => {
  vi.clearAllMocks()
})

describe('useRefinement only ever exposes boxes of the requested photo', () => {
  it('never renders the previous photo\'s boxes once another photo is requested', async () => {
    vi.mocked(api.getAnnotations).mockImplementation(id => id === 'A'
      ? Promise.resolve(annotation('A', [box('a1', 100), box('a2', 900)])) : new Promise(() => {}))
    const log: Render[] = []
    const view = render(<Probe imageId="A" jobId="det-A" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['a1', 'a2']))
    view.rerender(<Probe imageId="B" jobId="det-B" done log={log} />)
    const forB = log.filter(entry => entry.imageId === 'B')
    expect(forB.length).toBeGreaterThan(0)
    expect(forB.every(entry => entry.ids.length === 0 && !entry.saved && entry.source === null)).toBe(true)
  })

  it('rejects detection boxes from a job that belongs to another photo', async () => {
    vi.mocked(api.getAnnotations).mockImplementation(notFound)
    vi.mocked(api.getDetectionBoxes).mockResolvedValue({ image_id: 'A', boxes: [box('box-0', 100)] })
    const log: Render[] = []
    render(<Probe imageId="B" jobId="det-A" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.loading).toBe(false))
    expect(api.getDetectionBoxes).toHaveBeenCalledWith('det-A')
    expect(log.every(entry => entry.ids.length === 0)).toBe(true)
    expect(log.at(-1)?.error).toMatch(/drugoj fotografiji/)
  })

  it('ignores a saved annotation file written for another photo and falls back to this photo\'s detections', async () => {
    vi.mocked(api.getAnnotations).mockResolvedValue(annotation('A', [box('a1', 100)]))
    vi.mocked(api.getDetectionBoxes).mockResolvedValue({ image_id: 'B', boxes: [box('box-0', 400)] })
    const log: Render[] = []
    render(<Probe imageId="B" jobId="det-B" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0']))
    expect(log.every(entry => !entry.ids.includes('a1') && !entry.saved)).toBe(true)
  })

  it('keeps detection boxes of the same photo', async () => {
    vi.mocked(api.getAnnotations).mockImplementation(notFound)
    vi.mocked(api.getDetectionBoxes).mockResolvedValue({ image_id: 'B', boxes: [box('box-0', 400)] })
    const log: Render[] = []
    render(<Probe imageId="B" jobId="det-B" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0']))
    expect(log.at(-1)?.error).toBeNull()
  })
})
