import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import WorkspacePage from './WorkspacePage'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useCalibrationStore } from '../store/calibrationStore'
import { useJobStore } from '../store/jobStore'
import { useProjectStore } from '../store/projectStore'
import type { AnnotatedBox, Job } from '../api/types'
import { CENTERLINE_METHOD } from '../api/types'
import * as api from '../api/client'

// user2's sequence on 2026-11-11: open a project photo -> contour source/run -> detection re-run fails -> boxes shown.
// Every frame the viewer renders is recorded as (displayed image, boxes); a box must never be drawn over another photo.

vi.mock('../api/client', () => ({
  runContours: vi.fn(), getJob: vi.fn(), getCalibration: vi.fn(), getProject: vi.fn(), getImage: vi.fn(),
  getContourSource: vi.fn(), getContourOverview: vi.fn(), runDetection: vi.fn(), manualCalibrate: vi.fn(),
  updateProjectImageJobs: vi.fn(), getAnnotations: vi.fn(), getDetectionBoxes: vi.fn(), saveAnnotations: vi.fn(),
  saveDetectorDataset: vi.fn(),
  imageUrl: (_id: string, name: string) => '/image/' + name,
  outputFileUrl: (id: string, name: string) => '/outputs/' + id + '/' + name,
}))
vi.mock('../hooks/useJob', async () => {
  const { useJobStore: store } = await import('../store/jobStore')
  return { useJobProgress: (id: string | null) => store(s => id ? s.jobs[id] : undefined) }
})
vi.mock('../components/ImageUploader', () => ({ default: () => <div>Image uploader</div> }))

const frames: { src: string; ids: string[] }[] = []
let viewerSrc = ''
vi.mock('../components/ImageViewer', () => ({
  default: ({ transformOverlay, src }: { transformOverlay?: React.ReactNode; src: string }) => {
    viewerSrc = src
    return <div data-testid="image-viewer" data-src={src}>{transformOverlay}</div>
  },
}))
vi.mock('../components/BboxOverlay', () => ({
  default: ({ boxes }: { boxes: AnnotatedBox[] }) => {
    frames.push({ src: viewerSrc, ids: boxes.map(b => b.id) })
    return <svg data-testid="boxes" data-ids={boxes.map(b => b.id).join(',')} />
  },
}))

const photoA = { image_id: 'imgA', filename: 'A.jpg', path: '/A.jpg', thumbnail_path: null, width: 10446, height: 10026 }
const photoB = { image_id: 'imgB', filename: 'B.jpg', path: '/B.jpg', thumbnail_path: null, width: 10493, height: 10365 }
const box = (id: string, x1: number, y1: number): AnnotatedBox => ({ id, x1, y1, x2: x1 + 55, y2: y1 + 30, conf: .9, status: 'accepted' })
const boxesOf: Record<string, AnnotatedBox[]> = {
  imgA: [box('a-1', 1200, 4100), box('a-2', 1210, 4105), box('a-3', 7000, 2000)],
  imgB: [box('b-1', 300, 5200), box('b-2', 5100, 900), box('b-3', 9800, 9700)],
}
const imageOfSrc: Record<string, string> = { '/image/A.jpg': 'imgA', '/image/B.jpg': 'imgB', 'blob:contourB': 'imgB', 'blob:contourB2': 'imgB' }

function job(id: string, type: Job['type'], status: Job['status'], result: Record<string, unknown> = {}, error: string | null = null): Job {
  return { id, type, status, progress: 1, message: '', error, created_at: '', started_at: '', completed_at: '', result }
}
const contour = (id: string) => job(id, 'contours', 'completed', {
  image_id: 'imgB', model_name: 'exp03_no_os', measurement_method: CENTERLINE_METHOD, calibration_um_per_px: id === 'contourB' ? 8.57 : null,
  csv_path: '/o/m.csv', excel_path: '/o/m.xlsx', contours_path: '/o/c.json', mask_records_path: '/o/r.json', num_organisms: 3,
  measurements: [], volume_summary: { available: false, count: 0, unit: null, mean: null, median: null, min: null, max: null, total: null },
  detection_job_id: 'detB',
})

beforeEach(() => {
  vi.clearAllMocks()
  frames.length = 0
  localStorage.clear()
  useWorkspaceStore.getState().reset()
  useCalibrationStore.getState().clear()
  useProjectStore.setState({ currentProjectId: 'p1', currentProjectName: 'Pendifin Čačinci' })
  useJobStore.setState({ jobs: {} })
  for (const j of [job('detA', 'detection', 'completed', { image_id: 'imgA', csv_path: '/o/a.csv' }),
                   job('detB', 'detection', 'completed', { image_id: 'imgB', csv_path: '/o/b.csv' }), contour('contourB')]) {
    useJobStore.getState().updateJob(j)
  }
  vi.mocked(api.getProject).mockResolvedValue({ id: 'p1', name: 'Pendifin Čačinci', images: [
    { image_id: 'imgA', filename: 'A.jpg', detection_job_id: 'detA', measurement_job_id: null, folder: null },
    { image_id: 'imgB', filename: 'B.jpg', detection_job_id: 'detB', measurement_job_id: 'contourB', folder: null },
  ] } as never)
  vi.mocked(api.getImage).mockImplementation(async id => (id === 'imgA' ? photoA : photoB) as never)
  vi.mocked(api.getAnnotations).mockImplementation(async id => ({ image_id: id, image_filename: `${id}.jpg`,
    source_job_id: id === 'imgA' ? 'detA' : 'detB', created_at: '', boxes: boxesOf[id] }))
  vi.mocked(api.getDetectionBoxes).mockImplementation(async id => ({ image_id: id === 'detA' ? 'imgA' : 'imgB', boxes: [] }))
  vi.mocked(api.getCalibration).mockRejectedValue(new Error('no calibration'))
  vi.mocked(api.getContourSource).mockImplementation(async id => ({ job_id: id === 'imgB' ? 'contourB' : null, measurement_method: null }))
  vi.mocked(api.getContourOverview).mockImplementation(async jobId => Object.assign(new Blob(['jpeg']), { jobId }))
  URL.createObjectURL = vi.fn((blob: Blob & { jobId?: string }) => `blob:${blob.jobId}`) as never
  URL.revokeObjectURL = vi.fn()
  vi.mocked(api.getJob).mockImplementation(async id => useJobStore.getState().jobs[id])
  vi.mocked(api.runContours).mockResolvedValue({ job_id: 'contourB2', status: 'pending', model_name: 'exp03_no_os' } as never)
  vi.mocked(api.runDetection).mockResolvedValue({ job_id: 'detFail', status: 'pending' })
  vi.mocked(api.updateProjectImageJobs).mockResolvedValue({} as never)
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ organisms: [] }) }))
})

function expectEveryFrameOnItsOwnPhoto() {
  expect(frames.length).toBeGreaterThan(0)
  const wrong = frames.filter(frame => frame.ids.some(id => `img${id[0].toUpperCase()}` !== imageOfSrc[frame.src]))
  expect(wrong).toEqual([])
}

describe('Workspace boxes always belong to the photo being displayed', () => {
  it('open photo -> next photo -> contour source/run -> failing detection re-run -> boxes shown', async () => {
    useWorkspaceStore.setState({ image: photoA, detectionJobId: 'detA', measureJobId: null })
    render(<MemoryRouter><WorkspacePage /></MemoryRouter>)
    await waitFor(() => expect(screen.getByTestId('boxes')).toHaveAttribute('data-ids', 'a-1,a-2,a-3'))
    expect(screen.getByTestId('image-viewer')).toHaveAttribute('data-src', '/image/A.jpg')

    await waitFor(() => expect(screen.getByTitle('Next image (→)')).toBeEnabled())
    fireEvent.click(screen.getByTitle('Next image (→)'))
    await waitFor(() => expect(screen.getByTestId('image-viewer')).toHaveAttribute('data-src', 'blob:contourB'))

    // Saved masks were measured at 8.57 µm/px but this photo has no confirmed scale: the page re-runs contours.
    await waitFor(() => expect(api.runContours).toHaveBeenCalledWith(expect.objectContaining({ image_id: 'imgB', source_contour_job_id: 'contourB' })), { timeout: 2500 })
    act(() => useJobStore.getState().updateJob(contour('contourB2')))
    await waitFor(() => expect(screen.getByTestId('image-viewer')).toHaveAttribute('data-src', 'blob:contourB2'))

    const view = screen.getAllByRole('combobox').find(e => Array.from((e as HTMLSelectElement).options).some(o => o.value === 'both'))!
    fireEvent.change(view, { target: { value: 'both' } })
    await waitFor(() => expect(screen.getByTestId('boxes')).toHaveAttribute('data-ids', 'b-1,b-2,b-3'))

    fireEvent.click(screen.getByRole('button', { name: 'Re-run Detection' }))
    // The standard run sends no thresholds: the backend's detection profile applies.
    await waitFor(() => expect(api.runDetection).toHaveBeenCalledWith({ image_id: 'imgB', device: '0' }))
    act(() => useJobStore.getState().updateJob(job('detFail', 'detection', 'failed', {}, "ValueError: Invalid CUDA 'device=0' requested.")))
    await waitFor(() => expect(screen.getByTestId('boxes')).toHaveAttribute('data-ids', 'b-1,b-2,b-3'))
    expect(screen.getByTestId('image-viewer')).toHaveAttribute('data-src', 'blob:contourB2')

    fireEvent.click(screen.getByTitle('Previous image (←)'))
    await waitFor(() => expect(screen.getByTestId('boxes')).toHaveAttribute('data-ids', 'a-1,a-2,a-3'))
    expect(screen.getByTestId('image-viewer')).toHaveAttribute('data-src', '/image/A.jpg')
    expectEveryFrameOnItsOwnPhoto()
  })

  it('a photo opened with a detection job of another photo shows no boxes from that job', async () => {
    useWorkspaceStore.setState({ image: photoB, detectionJobId: 'detA', measureJobId: null })
    vi.mocked(api.getAnnotations).mockRejectedValue(new Error('No annotations found for this image'))
    vi.mocked(api.getDetectionBoxes).mockResolvedValue({ image_id: 'imgA', boxes: boxesOf.imgA })
    render(<MemoryRouter><WorkspacePage /></MemoryRouter>)
    await waitFor(() => expect(api.getDetectionBoxes).toHaveBeenCalledWith('detA'))
    await act(async () => { await Promise.resolve() })
    expect(screen.queryByTestId('boxes')).not.toBeInTheDocument()
    expectEveryFrameOnItsOwnPhotoOrNone()
  })
})

function expectEveryFrameOnItsOwnPhotoOrNone() {
  const wrong = frames.filter(frame => frame.ids.some(id => `img${id[0].toUpperCase()}` !== imageOfSrc[frame.src]))
  expect(wrong).toEqual([])
}
