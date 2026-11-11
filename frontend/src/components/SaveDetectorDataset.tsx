import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { saveDetectorDataset } from '../api/client'
import type { AnnotatedBox } from '../api/types'

export default function SaveDetectorDataset({ imageId, sourceJobId, boxes, disabled }: {
  imageId: string; sourceJobId: string | null; boxes: AnnotatedBox[]; disabled: boolean
}) {
  const [source, setSource] = useState('')
  const [busy, setBusy] = useState(false)
  const lock = useRef(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<{ version: string; signature: string } | null>(null)
  const signature = JSON.stringify([imageId, sourceJobId, boxes, source])
  async function save() {
    if (lock.current) return
    lock.current = true; setBusy(true); setError(null)
    try {
      const result = await saveDetectorDataset({ image_id: imageId, source_job_id: sourceJobId, boxes, source_description: source })
      setSaved({ version: result.version_id, signature })
    } catch (e) { setError(e instanceof Error ? e.message : 'Spremanje nije uspjelo.') }
    finally { lock.current = false; setBusy(false) }
  }
  return <section className="border-t p-4 space-y-2 text-xs" aria-label="Dataset za budući trening">
    <h3 className="font-semibold">Ispravljene detekcije za budući trening</h3>
    <p>Spremit će se original i {boxes.filter(b => b.status !== 'rejected').length} konačnih pravokutnika. Prazan skup sprema se kao pregledana negativna fotografija.</p>
    <label className="block">Izvor fotografije / bilješka
      <textarea className="border rounded w-full p-2 mt-1" maxLength={2000} value={source} onChange={e => setSource(e.target.value)} placeholder="Npr. pokus, ploča, datum snimanja…" />
    </label>
    <button className="w-full rounded bg-teal-700 text-white px-2 py-2 disabled:opacity-50" disabled={disabled || busy} onClick={() => void save()}>
      {busy ? 'Provjera skupova i spremanje…' : 'Spremi za budući trening'}
    </button>
    <p className="text-gray-500">Verzionirana kopija; postojeći skupovi i model se ne mijenjaju.</p>
    {error && <p role="alert" className="text-red-700 break-words">{error}</p>}
    {saved && saved.signature === signature && <p role="status" className="text-teal-800">Spremljeno: <Link className="underline" to={`/detector-dataset?version=${saved.version}`}>{saved.version}</Link></p>}
    <Link className="block text-blue-700 underline" to="/detector-dataset">Pregled i YOLO izvoz dataseta</Link>
  </section>
}
