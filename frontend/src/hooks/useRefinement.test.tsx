import { act, render, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useRefinement } from './useRefinement'
import type { AnnotatedBox } from '../api/types'
import * as api from '../api/client'

vi.mock('../api/client', () => ({ getAnnotations: vi.fn(), getDetectionBoxes: vi.fn(), saveAnnotations: vi.fn(), mergeDetections: vi.fn() }))

const box = (id: string, x1: number): AnnotatedBox => ({ id, x1, y1: 10, x2: x1 + 50, y2: 60, conf: .9, status: 'accepted' })
const annotation = (imageId: string, boxes: AnnotatedBox[]) => ({
  image_id: imageId, image_filename: `${imageId}.jpg`, source_job_id: `det-${imageId}`, created_at: '', boxes })

type Hook = ReturnType<typeof useRefinement>
type Render = { imageId: string | null; ids: string[]; saved: boolean; source: string | null; error: string | null; loading: boolean;
  canMerge: boolean; merge: Hook['mergeDetections']; unsaved: boolean; save: Hook['saveAnnotations']; toggle: Hook['toggleBox'] }

function Probe({ imageId, jobId, done, log }: { imageId: string | null; jobId: string | null; done: boolean; log: Render[] }) {
  const refinement = useRefinement(imageId, jobId, done)
  log.push({ imageId, ids: refinement.boxes.map(b => b.id), saved: refinement.annotationsSaved,
    source: refinement.restoredSourceJobId, error: refinement.error, loading: refinement.isLoading,
    canMerge: refinement.canMergeDetections, merge: refinement.mergeDetections, unsaved: refinement.hasUnsavedEdits,
    save: refinement.saveAnnotations, toggle: refinement.toggleBox })
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

describe('a newer detection run is not hidden by annotation files nobody reviewed', () => {
  const unreviewed = (imageId: string, source: string, boxes: AnnotatedBox[]) =>
    ({ ...annotation(imageId, boxes), source_job_id: source })
  const reviewed = (imageId: string, source: string, boxes: AnnotatedBox[]) =>
    ({ ...unreviewed(imageId, source, boxes), saved_at: '2026-11-11T13:00:00+00:00' })

  it('shows the current job\'s boxes instead of a file written from an older job at import', async () => {
    vi.mocked(api.getAnnotations).mockResolvedValue(unreviewed('P', 'legacy-job', [box('box-0', 100)]))
    vi.mocked(api.getDetectionBoxes).mockResolvedValue({ image_id: 'P', boxes: [box('box-0', 100), box('box-1', 700)] })
    const log: Render[] = []
    render(<Probe imageId="P" jobId="new-job" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0', 'box-1']))
    expect(api.getDetectionBoxes).toHaveBeenCalledWith('new-job')
    expect(log.at(-1)).toMatchObject({ saved: false, source: null, canMerge: false })
  })

  it('keeps the unreviewed file while the new run is pending or if it failed', async () => {
    vi.mocked(api.getAnnotations).mockResolvedValue(unreviewed('P', 'legacy-job', [box('box-0', 100)]))
    const log: Render[] = []
    render(<Probe imageId="P" jobId="new-job" done={false} log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0']))
    expect(api.getDetectionBoxes).not.toHaveBeenCalled()
  })

  it('still shows an unreviewed file for its own detection job', async () => {
    vi.mocked(api.getAnnotations).mockResolvedValue(unreviewed('P', 'legacy-job', [box('box-0', 100)]))
    const log: Render[] = []
    render(<Probe imageId="P" jobId="legacy-job" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0']))
    expect(api.getDetectionBoxes).not.toHaveBeenCalled()
  })

  it('keeps reviewed annotations and merges the new run on request, keeping added and rejected boxes', async () => {
    vi.mocked(api.getAnnotations).mockResolvedValue(reviewed('P', 'legacy-job',
      [box('box-0', 100), { ...box('box-1', 300), status: 'rejected' }, { ...box('added-1', 500), status: 'added' }]))
    const merged = [box('box-0', 102), box('box-7', 900), { ...box('prev-box-1', 300), status: 'rejected' as const },
      { ...box('added-1', 500), status: 'added' as const }]
    vi.mocked(api.mergeDetections).mockResolvedValue({ image_id: 'P', source_job_id: 'new-job', previous_source_job_id: 'legacy-job',
      boxes: merged, summary: { human_saved: true, new_detections: 3, new_kept: 2, new_dropped_on_rejected: 1, new_dropped_on_added: 0, new_dropped_inside_previous: 0,
        previous_accepted: 1, previous_accepted_kept: 0, added_kept: 1, rejected_kept: 1 } })
    const log: Render[] = []
    render(<Probe imageId="P" jobId="new-job" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0', 'box-1', 'added-1']))
    expect(log.at(-1)).toMatchObject({ saved: true, source: 'legacy-job', canMerge: true })
    expect(api.getDetectionBoxes).not.toHaveBeenCalled()

    let summary: Awaited<ReturnType<Hook['mergeDetections']>> = null
    await act(async () => { summary = await log.at(-1)!.merge() })
    expect(api.mergeDetections).toHaveBeenCalledWith('P', 'new-job')
    expect(summary).toMatchObject({ new_kept: 2, rejected_kept: 1 })
    expect(log.at(-1)).toMatchObject({ ids: ['box-0', 'box-7', 'prev-box-1', 'added-1'], saved: false, source: 'new-job', canMerge: false })
    // The merge is a preview: measuring must wait until it is saved.
    expect(log.at(-1)?.unsaved).toBe(true)
    vi.mocked(api.saveAnnotations).mockResolvedValue(undefined as never)
    await act(async () => { await log.at(-1)!.save('P', 'P.jpg', 'new-job') })
    expect(log.at(-1)).toMatchObject({ saved: true, unsaved: false })
  })

  it('marks box edits as unsaved until they are saved; a loaded file or job is not an edit', async () => {
    vi.mocked(api.getAnnotations).mockImplementation(notFound)
    vi.mocked(api.getDetectionBoxes).mockResolvedValue({ image_id: 'Q', boxes: [box('box-0', 100)] })
    const log: Render[] = []
    render(<Probe imageId="Q" jobId="det-Q" done log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0']))
    expect(log.at(-1)?.unsaved).toBe(false)
    act(() => { log.at(-1)!.toggle('box-0') })
    expect(log.at(-1)?.unsaved).toBe(true)
  })

  it('offers no merge while the detection job is still running', async () => {
    vi.mocked(api.getAnnotations).mockResolvedValue(reviewed('P', 'legacy-job', [box('box-0', 100)]))
    const log: Render[] = []
    render(<Probe imageId="P" jobId="new-job" done={false} log={log} />)
    await waitFor(() => expect(log.at(-1)?.ids).toEqual(['box-0']))
    expect(log.at(-1)?.canMerge).toBe(false)
  })
})
