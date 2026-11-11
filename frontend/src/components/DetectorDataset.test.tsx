import { act, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import SaveDetectorDataset from './SaveDetectorDataset'
import BoxCoordinateEditor from './BoxCoordinateEditor'
import { saveDetectorDataset } from '../api/client'

vi.mock('../api/client', () => ({ saveDetectorDataset: vi.fn() }))
const box = { id: 'box-0', x1: 10, y1: 10, x2: 30, y2: 40, conf: .9, status: 'accepted' as const }

describe('Detector dataset collection', () => {
  it('sends current unsaved boxes once and links to the resulting immutable version', async () => {
    let finish!: (value: { version_id: string; entry_id: string; revision: number; unchanged: boolean }) => void
    vi.mocked(saveDetectorDataset).mockImplementation(() => new Promise(resolve => { finish = resolve }))
    render(<MemoryRouter><SaveDetectorDataset imageId="photo" sourceJobId="original-job" boxes={[{ ...box, x2: 35 }]} disabled={false} /></MemoryRouter>)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Plate A, 2026-11-05' } })
    const button = screen.getByRole('button', { name: 'Spremi za budući trening' })
    fireEvent.click(button); fireEvent.click(button)
    expect(saveDetectorDataset).toHaveBeenCalledTimes(1)
    expect(saveDetectorDataset).toHaveBeenCalledWith({ image_id: 'photo', source_job_id: 'original-job', source_description: 'Plate A, 2026-11-05', boxes: [{ ...box, x2: 35 }] })
    expect(button).toBeDisabled()
    await act(async () => finish({ version_id: 'v000001-123456789abc', entry_id: 'entry', revision: 1, unchanged: false }))
    expect(screen.getByRole('status')).toHaveTextContent('v000001-123456789abc')
  })

  it('keeps a failed validation visible and allows retry without losing the source note', async () => {
    vi.mocked(saveDetectorDataset).mockRejectedValueOnce(new Error('Fotografija je već u test skupu.'))
    render(<MemoryRouter><SaveDetectorDataset imageId="photo" sourceJobId={null} boxes={[]} disabled={false} /></MemoryRouter>)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Keep my note' } })
    fireEvent.click(screen.getByRole('button', { name: 'Spremi za budući trening' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('već u test skupu')
    expect(screen.getByRole('textbox')).toHaveValue('Keep my note')
    expect(screen.getByRole('button', { name: 'Spremi za budući trening' })).toBeEnabled()
  })

  it('validates numeric coordinate edits and preserves the original detection ID', () => {
    const apply = vi.fn()
    render(<BoxCoordinateEditor box={box} width={100} height={80} onApply={apply} />)
    fireEvent.change(screen.getByRole('spinbutton', { name: 'x2' }), { target: { value: '101' } })
    expect(screen.getByRole('button', { name: 'Primijeni koordinate' })).toBeDisabled()
    fireEvent.change(screen.getByRole('spinbutton', { name: 'x2' }), { target: { value: '35' } })
    fireEvent.click(screen.getByRole('button', { name: 'Primijeni koordinate' }))
    expect(apply).toHaveBeenCalledWith('box-0', { x1: 10, y1: 10, x2: 35, y2: 40 })
  })
})
