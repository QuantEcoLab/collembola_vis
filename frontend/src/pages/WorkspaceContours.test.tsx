import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import WorkspacePage from './WorkspacePage'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useCalibrationStore } from '../store/calibrationStore'
import { useJobStore } from '../store/jobStore'
import { useProjectStore } from '../store/projectStore'
import type { Job } from '../api/types'
import * as api from '../api/client'

vi.mock('../api/client', () => ({
  runContours: vi.fn(), getJob: vi.fn(), getCalibration: vi.fn(), getProject: vi.fn(), getImage: vi.fn(),
  runDetection: vi.fn(), runMeasurement: vi.fn(), manualCalibrate: vi.fn(), updateProjectImageJobs: vi.fn(),
  imageUrl: (_id: string, name: string) => '/image/' + name,
  outputFileUrl: (id: string, name: string) => '/outputs/' + id + '/' + name,
}))
vi.mock('../hooks/useJob', async () => {
  const { useJobStore: store } = await import('../store/jobStore')
  return { useJobProgress: (id: string | null) => store(s => id ? s.jobs[id] : undefined) }
})
vi.mock('../hooks/useRefinement', () => ({ useRefinement: () => ({ boxes: [], selectedId: null, annotationsSaved: false,
  restoredSourceJobId: null, selectBox: vi.fn(), toggleBox: vi.fn(), removeBox: vi.fn(), saveAnnotations: vi.fn(),
  acceptedCount: 0, rejectedCount: 0, addedCount: 0, drawingBox: null,
}) }))
vi.mock('../components/ImageUploader', () => ({ default: () => <div>Image uploader</div> }))
vi.mock('../components/ImageViewer', () => ({ default: ({ transformOverlay }: { transformOverlay?: React.ReactNode }) => <div data-testid="image-viewer">{transformOverlay}</div> }))

const photo = { image_id: 'photo1', filename: 'photo.png', path: '/photo.png', thumbnail_path: null, width: 100, height: 80 }
const measurement = { detection_id: 17, segmentation_confidence: .95, volume_mm3: null, volume_unit: 'nije dostupan bez kalibracije',
  length_px: 20, width_px: 5, area_px: 70, perimeter_px: 40, calibration_um_per_px: null,
  dedup_status: 'kept', touches_crop_edge: false, model_name: 'exp03_no_os' }
function completed(id = 'contour1', calibrated = false): Job {
  return { id, type: 'contours', status: 'completed', progress: 1, message: 'Done', error: null,
    created_at: '', started_at: '', completed_at: '', result: {
      image_id: photo.image_id, model_name: 'exp03_no_os', csv_path: '/output/measures.csv',
      excel_path: '/output/measures.xlsx', overlay_path: '/output/overlay.jpg', contours_path: '/output/contours.json',
      calibration_um_per_px: calibrated ? 10 : null, num_organisms: 1,
      measurements: [{ ...measurement, volume_mm3: calibrated ? .00003 : null, calibration_um_per_px: calibrated ? 10 : null }],
      volume_summary: { available: calibrated, count: calibrated ? 1 : 0, unit: calibrated ? 'mm³' : null,
        mean: calibrated ? .00003 : null, median: calibrated ? .00003 : null, min: calibrated ? .00003 : null,
        max: calibrated ? .00003 : null, total: calibrated ? .00003 : null },
    } }
}
function renderPage() { return render(<MemoryRouter><WorkspacePage /></MemoryRouter>) }

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  useWorkspaceStore.getState().reset()
  useCalibrationStore.getState().clear()
  useJobStore.setState({ jobs: {} })
  useProjectStore.setState({ currentProjectId: null, currentProjectName: null })
  vi.mocked(api.getCalibration).mockRejectedValue(new Error('no image calibration'))
  vi.mocked(api.getJob).mockResolvedValue(completed())
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ organisms: [
    { detection_id: 17, bbox: [5, 5, 25, 45], contours: [[[5, 5], [25, 5], [25, 45]]] },
  ] }) }))
})

describe('Workspace contour request and result workflow', () => {
  it('reports missing image without making a backend request', () => {
    renderPage()
    fireEvent.click(screen.getByRole('button', { name: 'Napravi konture' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Najprije učitajte fotografiju')
    expect(api.runContours).not.toHaveBeenCalled()
  })

  it('posts only image and confirmed calibration; ignores the old global default and duplicate clicks', async () => {
    useWorkspaceStore.getState().setImage(photo)
    useCalibrationStore.getState().setUmManual(8.57) // Belongs to a previous image, not proof for this one.
    let finish!: (value: { job_id: string; status: string; model_name: string }) => void
    vi.mocked(api.runContours).mockImplementation(() => new Promise(resolve => { finish = resolve }))
    renderPage()
    const button = screen.getByRole('button', { name: 'Napravi konture' })
    fireEvent.click(button)
    fireEvent.click(button)
    expect(api.runContours).toHaveBeenCalledTimes(1)
    expect(api.runContours).toHaveBeenCalledWith({ image_id: photo.image_id, um_per_pixel: null })
    expect(button).toBeDisabled()
    await act(async () => { finish({ job_id: 'contour1', status: 'pending', model_name: 'exp03_no_os' }) })
    act(() => useJobStore.getState().updateJob({ ...completed(), status: 'failed', error: 'Segmentation failed' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Segmentation failed'))
    expect(screen.getByRole('button', { name: 'Napravi konture' })).toBeEnabled()
  })

  it('loads automatic measurement results and highlights the correct detection ID rather than its row index', async () => {
    useWorkspaceStore.setState({ image: photo, contourJobId: 'contour1' })
    useJobStore.getState().updateJob(completed())
    renderPage()
    await waitFor(() => expect(screen.getByTestId('volume-summary')).toHaveTextContent('Konačnih jedinki s maskom: 1'))
    fireEvent.click(screen.getByRole('button', { name: /View Results/ }))
    fireEvent.click(await screen.findByText('17'))
    await waitFor(() => expect(screen.getAllByTestId('selected-contour').every(element => element.getAttribute('data-detection-id') === '17')).toBe(true))
    expect(screen.getByText('nije dostupan bez kalibracije')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Excel' })).toBeEnabled()
    fireEvent.click(screen.getByTitle('Close (Esc)'))
    const view = screen.getAllByRole('combobox').find(element => Array.from((element as HTMLSelectElement).options).some(option => option.value === 'raw'))!
    fireEvent.change(view, { target: { value: 'raw' } })
    await waitFor(() => expect(screen.queryByTestId('selected-contour')).not.toBeInTheDocument())
    expect(view).toHaveValue('raw')
    fireEvent.change(view, { target: { value: 'contours' } })
    await waitFor(() => expect(screen.getByTestId('selected-contour')).toHaveAttribute('data-detection-id', '17'))
    expect(api.runMeasurement).not.toHaveBeenCalled()
  })

  it('changing calibration requests measurement from saved masks, never a new detector run', async () => {
    useWorkspaceStore.setState({ image: photo, contourJobId: 'contour1' })
    useJobStore.getState().updateJob(completed())
    vi.mocked(api.runContours).mockResolvedValue({ job_id: 'recalibrated', status: 'pending', model_name: 'exp03_no_os' })
    renderPage()
    await waitFor(() => expect(screen.getByTestId('volume-summary')).toBeInTheDocument())
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '10' } })
    await waitFor(() => expect(api.runContours).toHaveBeenCalledWith({ image_id: photo.image_id, um_per_pixel: 10, source_contour_job_id: 'contour1' }), { timeout: 2500 })
    expect(api.runDetection).not.toHaveBeenCalled()
    act(() => useJobStore.getState().updateJob(completed('recalibrated', true)))
    await waitFor(() => expect(screen.getByTestId('volume-summary')).toHaveTextContent('10 µm/px'))
  })
})
