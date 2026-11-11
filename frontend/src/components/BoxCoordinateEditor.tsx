import { useState } from 'react'
import type { AnnotatedBox } from '../api/types'

export default function BoxCoordinateEditor({ box, width, height, onApply }: {
  box: AnnotatedBox; width: number; height: number
  onApply: (id: string, coords: { x1: number; y1: number; x2: number; y2: number }) => void
}) {
  const [coords, setCoords] = useState({ x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 })
  const valid = Object.values(coords).every(Number.isFinite) && coords.x1 >= 0 && coords.y1 >= 0 && coords.x2 <= width && coords.y2 <= height && coords.x1 < coords.x2 && coords.y1 < coords.y2
  return <form className="p-2 border rounded space-y-1 text-xs max-w-52" onSubmit={e => { e.preventDefault(); if (valid) onApply(box.id, coords) }}>
    <p className="font-medium truncate" title={box.id}>Ispravi pravokutnik: {box.id}</p>
    <div className="grid grid-cols-2 gap-1">{(['x1', 'y1', 'x2', 'y2'] as const).map(key => <label key={key}>{key} (px)
      <input type="number" step="any" aria-label={key} value={Number.isNaN(coords[key]) ? '' : coords[key]} className="border rounded w-full p-1" onChange={e => setCoords({ ...coords, [key]: e.target.valueAsNumber })} />
    </label>)}</div>
    {!valid && <p className="text-red-700">Pravokutnik mora biti neprazan i unutar slike.</p>}
    <button disabled={!valid} className="bg-amber-100 rounded p-1 disabled:opacity-50">Primijeni koordinate</button>
  </form>
}
