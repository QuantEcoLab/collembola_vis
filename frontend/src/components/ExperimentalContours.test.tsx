import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { WorkspaceSidebar } from './WorkspaceSidebar'
import ExperimentalContourAction from './ExperimentalContourAction'
import MeasurementTable from './MeasurementTable'
import VolumeSummary from './VolumeSummary'
import { CENTERLINE_METHOD } from '../api/types'

describe('Experimental contour controls and measurement presentation', () => {
  it('keeps the new button after every original left-column control, including Reset', () => {
    const { container } = render(<WorkspaceSidebar
      imageFilename="photo.jpg" imageWidth={100} imageHeight={80} umPerPixel={8.57} setUmPerPixel={vi.fn()}
      calibrated={false} calibrationMode={false} calibrationPointCount={0} calibrationError={null} calibrationMessage={null}
      onStartCalibration={vi.fn()} onCancelCalibration={vi.fn()} detectionDone={true} detectionJob={null}
      detectionError={null} onRunDetection={vi.fn()} onAdvancedSettings={vi.fn()} annotationsSaved={false} boxCount={3}
      measurementDone={true} measureJob={null} measureError={null}
      onRunMeasurement={vi.fn()} csvDataLength={2} onViewResults={vi.fn()} onReset={vi.fn()}
      contourBusy={false} contourJob={null} contourError={null} contourCalibration={null} onRunContours={vi.fn()}
      volumeSummary={null} contourCount={0}
    />)
    const buttons = Array.from(container.querySelectorAll('button'))
    expect(buttons.at(-1)).toHaveTextContent('Napravi konture')
    const reset = screen.getByRole('button', { name: /Reset Workspace/ }) as HTMLButtonElement
    expect(buttons.indexOf(reset)).toBeLessThan(buttons.length - 1)
    expect(screen.getByRole('button', { name: /Advanced Settings/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /View Results/ })).toBeInTheDocument()
    expect(screen.queryByText('Fast Ellipse')).not.toBeInTheDocument()
  })

  it('disables repeat clicks and keeps errors above the final button', () => {
    const run = vi.fn()
    const { container } = render(<ExperimentalContourAction busy={true} job={null} error="Neuspjela obrada" calibration={null} onRun={run} />)
    const button = screen.getByRole('button')
    expect(button).toBeDisabled()
    fireEvent.click(button)
    expect(run).not.toHaveBeenCalled()
    expect(container.firstElementChild?.lastElementChild).toBe(button)
    expect(screen.getByRole('alert')).toHaveTextContent('Neuspjela obrada')
  })

  it('shows pixel measurements and unavailable volume without calibration', () => {
    render(<MeasurementTable data={[{ detection_id: 17, segmentation_confidence: .9, volume_mm3: null, volume_unit: 'unavailable',
      length_px: 20, width_px: 6, area_px: 100, perimeter_px: 45, length_mm: null, width_mm: null,
      calibration_um_per_px: null, dedup_status: 'kept', touches_crop_edge: true, warning: 'Rub', model_name: 'exp03_no_os', measurement_method: CENTERLINE_METHOD }]} />)
    expect(screen.getByText('Zakrivljena duljina (px)')).toBeInTheDocument()
    expect(screen.queryByText('Zakrivljena duljina (mm)')).not.toBeInTheDocument()
    expect(screen.getByText('nije dostupan bez kalibracije')).toBeInTheDocument()
    expect(screen.getByText('DA')).toBeInTheDocument()
  })

  it('emphasizes calibrated volume and maps row selection to the original data index after sorting', () => {
    const click = vi.fn()
    render(<MeasurementTable onRowClick={click} data={[
      { detection_id: 17, segmentation_confidence: .9, volume_mm3: .00000234, calibration_um_per_px: 8.57, model_name: 'exp03_no_os', measurement_method: CENTERLINE_METHOD },
      { detection_id: 4, segmentation_confidence: .9, volume_mm3: .00000123, calibration_um_per_px: 8.57, model_name: 'exp03_no_os', measurement_method: CENTERLINE_METHOD },
    ]} />)
    expect(screen.getByText('0.00000234').closest('td')).toHaveClass('font-bold')
    fireEvent.click(screen.getByText('Detection ID'))
    fireEvent.click(screen.getByText('Detection ID'))
    fireEvent.click(screen.getByText('4').closest('tr')!)
    expect(click).toHaveBeenCalledWith(1)
  })

  it('has all five volume summary statistics and no fake total when uncalibrated', () => {
    render(<VolumeSummary count={2} calibration={null} summary={{ available: false, count: 0, unit: null, mean: null, median: null, min: null, max: null, total: null }} />)
    expect(screen.getByText('Volumen nije dostupan bez kalibracije. Unesite µm/px, odaberite poznati preset ili kalibrirajte ravnalom.')).toBeInTheDocument()
    expect(screen.getByText('Srednji')).toBeInTheDocument()
    expect(screen.getByText('Medijan')).toBeInTheDocument()
    expect(screen.getByText('Minimalni')).toBeInTheDocument()
    expect(screen.getByText('Maksimalni')).toBeInTheDocument()
    expect(screen.getByText('Ukupni procijenjeni volumen pouzdanih jedinki')).toBeInTheDocument()
    expect(screen.queryByText('0 mm³')).not.toBeInTheDocument()
  })

  it('does not render archived-method values and clearly marks an excluded current-method candidate', () => {
    const { rerender } = render(<MeasurementTable data={[{ volume_mm3: 98765, model_name: 'exp03_no_os', length_mm: 321 }]} />)
    expect(screen.queryByText('98765')).not.toBeInTheDocument()
    expect(screen.getByText(/pokrenite mjerenje središnjom linijom/)).toBeInTheDocument()
    rerender(<MeasurementTable data={[{ detection_id: 8, volume_mm3: .25, calibration_um_per_px: 8,
      measurement_method: CENTERLINE_METHOD, measurement_reliable: false, measurement_status: 'nepouzdano',
      volume_included: false, warning: 'Maska dodiruje rub cropa' }]} />)
    expect(screen.getByText('nepouzdano').closest('tr')).toHaveClass('bg-orange-50')
    expect(screen.getByText('NE')).toBeInTheDocument()
    expect(screen.getByText('U ukupnom volumenu')).toBeInTheDocument()
  })
})
