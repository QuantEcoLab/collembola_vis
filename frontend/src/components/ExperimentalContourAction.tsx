import { Loader2, ScanLine } from 'lucide-react'
import JobProgress from './JobProgress'
import type { Job } from '../api/types'

interface Props {
  busy: boolean
  job: Job | null
  error: string | null
  calibration: number | null
  onRun: () => void
}

export default function ExperimentalContourAction({ busy, job, error, calibration, onRun }: Props) {
  return (
    <div data-testid="experimental-contour-action" className="shrink-0 px-4 pb-4 pt-3 border-t border-violet-200 bg-violet-50">
      <p className="text-xs font-semibold text-violet-900">Eksperimentalni model exp03_no_os</p>
      <p className="text-[11px] text-violet-800 mt-1">Konture → postojeći dedup → automatsko mjerenje. Izlaz modela nije GT oznaka.</p>
      {calibration == null && <p className="text-[11px] text-amber-800 mt-1">Kalibracija za ovu fotografiju nije potvrđena. Mjere će biti u pikselima; volumen nije dostupan bez kalibracije.</p>}
      {busy && job && <JobProgress job={job} />}
      {busy && !job && <p role="status" className="text-xs mt-2">Pokretanje obrade…</p>}
      {error && <p role="alert" className="text-xs text-red-700 mt-2">{error}</p>}
      <button
        data-testid="make-contours-button"
        onClick={onRun}
        disabled={busy}
        aria-busy={busy}
        className="w-full mt-3 px-3 py-2 rounded-lg bg-violet-700 text-white font-semibold text-sm hover:bg-violet-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {busy ? <Loader2 size={15} className="animate-spin" /> : <ScanLine size={15} />}
        {busy ? 'Obrada kontura i mjerenja…' : 'Napravi konture'}
      </button>
    </div>
  )
}
