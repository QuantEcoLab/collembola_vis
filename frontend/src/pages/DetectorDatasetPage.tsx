import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { detectorDatasetBlob, getDetectorDatasetVersion, listDetectorDataset } from '../api/client'
import type { DetectorDatasetIndex, DetectorDatasetVersion } from '../api/types'
import BboxOverlay from '../components/BboxOverlay'

const actionLabel = { added: 'Dodan', corrected: 'Ispravljen', confirmed: 'Potvrđen' }

export default function DetectorDatasetPage() {
  const [params, setParams] = useSearchParams()
  const version = params.get('version') || ''
  const [index, setIndex] = useState<DetectorDatasetIndex | null>(null)
  const [detail, setDetail] = useState<DetectorDatasetVersion | null>(null)
  const [entryId, setEntryId] = useState('')
  const [selectedBox, setSelectedBox] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    let cancelled = false
    setLoading(true); setError(null)
    listDetectorDataset().then(data => {
      if (cancelled) return
      setIndex(data)
      if (!version && data.versions.length) setParams({ version: data.versions[0].version_id }, { replace: true })
    }).catch(e => { if (!cancelled) setError(e.message) }).finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [retry])
  useEffect(() => {
    let cancelled = false
    setDetail(null); setError(null)
    if (version) getDetectorDatasetVersion(version).then(data => {
      if (!cancelled) { setDetail(data); setEntryId(data.entries[0]?.entry_id || '') }
    }).catch(e => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [version, retry])
  useEffect(() => {
    const controller = new AbortController()
    let url = ''
    setImageUrl(''); setSelectedBox(null)
    if (version && entryId) detectorDatasetBlob(version, entryId, controller.signal).then(blob => {
      if (!controller.signal.aborted) { url = URL.createObjectURL(blob); setImageUrl(url) }
    }).catch(e => { if (!controller.signal.aborted) setError(e.message) })
    return () => { controller.abort(); if (url) URL.revokeObjectURL(url) }
  }, [version, entryId])
  const entry = detail?.entries.find(e => e.entry_id === entryId)
  async function download() {
    if (exporting) return
    setExporting(true); setError(null)
    try {
      const blob = await detectorDatasetBlob(version)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `detector-${version}-yolo.zip`; anchor.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (e) { setError(e instanceof Error ? e.message : 'Izvoz nije uspio.') }
    finally { setExporting(false) }
  }
  return <div className="min-h-screen bg-gray-50 pb-32">
    <header className="bg-white border-b p-4 flex gap-6"><Link to="/projects">Projects</Link><Link to="/workspace">Workspace</Link><strong>Dataset za budući trening detektora</strong></header>
    <main className="max-w-7xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Ispravljene detekcije</h1>
      <p>Zasebne, nepromjenjive verzije originalnih fotografija i konačnih pravokutnika. Ponovno spremanje iste fotografije stvara reviziju, a ne duplikat u istoj verziji.</p>
      <p className="text-sm text-gray-600">YOLO izvoz: klasa 0 (collembola), tileovi 1280 / preklop 256, slike i normalizirane bbox oznake. Skup je namijenjen budućem treningu; validation/test se ne dodjeljuju automatski. Za budući trening treba zasebno postaviti neovisan validation skup.</p>
      {index?.registry ? <details className="text-xs"><summary>Opseg provjere skupova: {index.registry.sources} izvora, {index.registry.images} slika · {new Date(index.registry.created_at).toLocaleString()}</summary>{index.registry.roots.map(r => <p key={r} className="break-all">{r}</p>)}</details> : !loading && <p className="text-amber-800">Registar postojećih skupova nije spreman; spremanje je blokirano.</p>}
      {error && <div role="alert" className="bg-red-50 border border-red-200 p-3 text-red-800">{error} <button onClick={() => setRetry(r => r + 1)} className="underline">Pokušaj ponovno</button></div>}
      {loading ? <p>Učitavanje verzija…</p> : index?.versions.length === 0 ? <p>Još nema spremljenih fotografija. U Workspaceu uredite pravokutnike i odaberite „Spremi za budući trening”.</p> : <label>Verzija <select className="border rounded p-2" value={version} onChange={e => setParams({ version: e.target.value })}>{index?.versions.map(v => <option key={v.version_id} value={v.version_id}>{v.version_id} · {v.image_count} fotografija · {v.box_count} pravokutnika · {v.created_by}</option>)}</select></label>}
      {detail && <>
        <button disabled={exporting} onClick={() => void download()} className="rounded bg-teal-700 text-white px-4 py-2 disabled:opacity-50">{exporting ? 'Priprema YOLO izvoza…' : 'Izvezi YOLO ZIP'}</button>
        <p className="text-xs">Spremljeno: {new Date(detail.created_at).toLocaleString()} · korisnik {detail.created_by} · prethodna verzija {detail.parent_version || '—'}</p>
        <label className="block">Fotografija <select className="border p-2" value={entryId} onChange={e => setEntryId(e.target.value)}>{detail.entries.map(e => <option value={e.entry_id} key={e.entry_id}>{e.filename} · revizija {e.revision} · {e.boxes.length} pravokutnika</option>)}</select></label>
      </>}
      {entry && <>
        <div className="bg-white border rounded p-3 text-sm space-y-1">
          <p>{entry.width} × {entry.height} px · {entry.saved_by} · {new Date(entry.saved_at).toLocaleString()}</p>
          <p>Izvor: {entry.source.description || entry.source.resolved_path} ({entry.source.ingestion})</p>
          <p>Dodani: {entry.change_counts.added} · ispravljeni: {entry.change_counts.corrected} · potvrđeni: {entry.change_counts.confirmed} · uklonjeni: {entry.removed_detections.length}</p>
          <details><summary>Uklonjene izvorne detekcije</summary><p className="break-all text-xs">{entry.removed_detections.map(b => b.id).join(', ') || 'Nema'}</p></details>
          <p className="break-all text-xs">SHA-256 originala: {entry.image_sha256}</p>
        </div>
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="relative self-start border bg-white">{imageUrl ? <><img className="w-full block" src={imageUrl} alt={entry.filename} /><BboxOverlay boxes={entry.boxes} imageWidth={entry.width} imageHeight={entry.height} mode="review" selectedId={selectedBox} onBoxClick={setSelectedBox} /></> : <p>Učitavanje originala…</p>}</div>
          <div className="max-h-[650px] overflow-auto bg-white border"><table className="text-xs w-full"><thead><tr><th>ID</th><th>Izmjena</th><th>x1, y1 → x2, y2</th></tr></thead><tbody>{entry.boxes.map(b => <tr key={b.id} onClick={() => setSelectedBox(b.id)} className={`cursor-pointer border-t ${selectedBox === b.id ? 'bg-amber-100' : ''}`}><td className="p-2 max-w-36 truncate" title={b.id}>{b.id}</td><td>{actionLabel[b.action]}</td><td>{[b.x1, b.y1, b.x2, b.y2].map(v => v.toFixed(1)).join(', ')}{b.action === 'corrected' && b.original_bbox && <p className="text-gray-500">Prije: {['x1', 'y1', 'x2', 'y2'].map(k => b.original_bbox![k].toFixed(1)).join(', ')}</p>}</td></tr>)}</tbody></table></div>
        </div>
      </>}
    </main>
  </div>
}
