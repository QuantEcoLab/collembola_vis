import type { VolumeSummary as Summary } from '../api/types'

export function formatVolume(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : Number(value.toPrecision(6)).toString()
}

export default function VolumeSummary({ summary, count, calibration }: { summary: Summary; count: number; calibration: number | null }) {
  return (
    <section data-testid="volume-summary" className="rounded-lg border border-violet-200 bg-violet-50 p-3 text-xs">
      <h3 className="font-bold text-violet-950">Procijenjeni volumen</h3>
      <p className="mt-1">Konačnih jedinki s maskom: <strong>{count}</strong></p>
      <p className="text-[11px] mt-1">Zakrivljena središnja linija i lokalni kružni presjeci: V = Σ π × (wᵢ / 2)² × Δlᵢ.</p>
      <p className="mt-2">Pouzdanih: <strong>{summary.reliable_count ?? summary.count}</strong> · nepouzdanih / isključenih: <strong>{summary.excluded_count ?? 0}</strong></p>
      <p className="text-[11px] text-amber-900">Ukupni volumen i sve statistike uključuju samo pouzdane, kalibrirane jedinke.</p>
      {!summary.available && <p className="font-semibold text-amber-900 mt-2">Volumen nije dostupan bez kalibracije. Unesite µm/px, odaberite poznati preset ili kalibrirajte ravnalom.</p>}
      {calibration != null && <p className="mt-2">Korištena kalibracija: {calibration} µm/px</p>}
      <dl className="mt-2 space-y-1">
        {([['Srednji', summary.mean], ['Medijan', summary.median], ['Minimalni', summary.min], ['Maksimalni', summary.max]] as const).map(([label, value]) => (
          <div key={label} className="flex justify-between gap-2"><dt>{label}</dt><dd>{formatVolume(value)}{summary.available ? ' mm³' : ''}</dd></div>
        ))}
        <div className="border-t border-violet-200 pt-2 mt-2"><dt className="font-semibold">Ukupni procijenjeni volumen pouzdanih jedinki</dt><dd className="text-lg font-bold text-violet-950">{formatVolume(summary.total)}{summary.available ? ' mm³' : ''}</dd></div>
      </dl>
    </section>
  )
}
