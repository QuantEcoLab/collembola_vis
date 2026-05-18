import { useRef, useState, useEffect, useCallback, type ReactNode } from 'react'
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react'

interface Props {
  src: string
  alt?: string
  className?: string
  onImageClick?: (x: number, y: number) => void
  overlay?: ReactNode
  /** SVG/overlay rendered inside the transform (moves with image). */
  transformOverlay?: ReactNode
  /** When true, panning is disabled (e.g. while drawing boxes). */
  disablePan?: boolean
}

interface Transform { scale: number; x: number; y: number }

const MIN_SCALE = 0.02
const MAX_SCALE = 40
const ZOOM_STEP = 1.15   // per wheel tick / button press

export default function ImageViewer({
  src,
  alt = 'Image',
  className = '',
  onImageClick,
  overlay,
  transformOverlay,
  disablePan,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const innerRef    = useRef<HTMLDivElement>(null)
  const imgRef      = useRef<HTMLImageElement>(null)

  // Transform is stored in a ref and applied directly to the DOM so that
  // every pointermove doesn't trigger a React re-render.
  const t = useRef<Transform>({ scale: 1, x: 0, y: 0 })
  const [loaded, setLoaded] = useState(false)

  // ── Apply transform to DOM ──────────────────────────────────────────────────
  const applyTransform = useCallback((next: Transform) => {
    t.current = next
    if (innerRef.current) {
      innerRef.current.style.transform =
        `translate(${next.x}px, ${next.y}px) scale(${next.scale})`
    }
  }, [])

  // ── Fit image inside container ──────────────────────────────────────────────
  const fitToContainer = useCallback(() => {
    const container = containerRef.current
    const img       = imgRef.current
    if (!container || !img || !img.naturalWidth) return
    const { width: cw, height: ch } = container.getBoundingClientRect()
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight)
    applyTransform({
      scale,
      x: (cw - img.naturalWidth  * scale) / 2,
      y: (ch - img.naturalHeight * scale) / 2,
    })
  }, [applyTransform])

  // ── Reset transform when source changes ────────────────────────────────────
  useEffect(() => { setLoaded(false) }, [src])

  // ── Wheel: zoom toward cursor ───────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const onWheel = (e: WheelEvent) => {
      e.preventDefault()

      // ctrlKey is set by browser for trackpad pinch-to-zoom gestures.
      // For plain trackpad scroll (no ctrl), pan instead of zoom.
      if (!e.ctrlKey && !e.metaKey && Math.abs(e.deltaY) < 60 && Math.abs(e.deltaX) > 0) {
        // Trackpad two-finger scroll → pan
        applyTransform({
          ...t.current,
          x: t.current.x - e.deltaX,
          y: t.current.y - e.deltaY,
        })
        return
      }

      const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP
      const rect  = container.getBoundingClientRect()
      const px    = e.clientX - rect.left
      const py    = e.clientY - rect.top
      const cur   = t.current
      const next  = Math.max(MIN_SCALE, Math.min(MAX_SCALE, cur.scale * factor))
      const ratio = next / cur.scale
      applyTransform({ scale: next, x: px - (px - cur.x) * ratio, y: py - (py - cur.y) * ratio })
    }

    container.addEventListener('wheel', onWheel, { passive: false })
    return () => container.removeEventListener('wheel', onWheel)
  }, [applyTransform])

  // ── Pan via left-drag ───────────────────────────────────────────────────────
  const pan = useRef({ active: false, moved: false, ox: 0, oy: 0 })

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (disablePan || e.button !== 0) {
      pan.current.moved = false
      return
    }
    pan.current = { active: true, moved: false, ox: e.clientX, oy: e.clientY }
    e.currentTarget.setPointerCapture(e.pointerId)
    containerRef.current!.style.cursor = 'grabbing'
  }, [disablePan])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!pan.current.active) return
    const dx = e.clientX - pan.current.ox
    const dy = e.clientY - pan.current.oy
    pan.current.ox = e.clientX
    pan.current.oy = e.clientY
    if (Math.abs(dx) + Math.abs(dy) > 2) pan.current.moved = true
    applyTransform({ ...t.current, x: t.current.x + dx, y: t.current.y + dy })
  }, [applyTransform])

  const onPointerUp = useCallback(() => {
    pan.current.active = false
    if (containerRef.current)
      containerRef.current.style.cursor = disablePan ? 'default' : 'grab'
  }, [disablePan])

  // ── Calibration click → image coordinates ─────────────────────────────────
  const onClick = useCallback((e: React.MouseEvent) => {
    if (!onImageClick || pan.current.moved) return
    const rect = containerRef.current!.getBoundingClientRect()
    const imgX = (e.clientX - rect.left  - t.current.x) / t.current.scale
    const imgY = (e.clientY - rect.top   - t.current.y) / t.current.scale
    onImageClick(imgX, imgY)
  }, [onImageClick])

  // ── Zoom buttons ───────────────────────────────────────────────────────────
  const zoomCenter = useCallback((factor: number) => {
    const container = containerRef.current
    if (!container) return
    const { width: cw, height: ch } = container.getBoundingClientRect()
    const px = cw / 2, py = ch / 2
    const cur   = t.current
    const next  = Math.max(MIN_SCALE, Math.min(MAX_SCALE, cur.scale * factor))
    const ratio = next / cur.scale
    applyTransform({ scale: next, x: px - (px - cur.x) * ratio, y: py - (py - cur.y) * ratio })
  }, [applyTransform])

  // Stop button events from bubbling into pan/click handlers
  const btnDown = (e: React.PointerEvent) => e.stopPropagation()
  const btnClick = (fn: () => void) => (e: React.MouseEvent) => { e.stopPropagation(); fn() }

  return (
    <div
      ref={containerRef}
      className={`relative bg-gray-100 overflow-hidden select-none ${className}`}
      style={{ cursor: disablePan ? 'default' : 'grab' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerUp}
      onPointerCancel={onPointerUp}
      onClick={onClick}
    >
      {/* ── Transformed image + overlay ─────────────────────────────────── */}
      <div
        ref={innerRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          transformOrigin: '0 0',
          transform: 'translate(0px,0px) scale(1)',
          visibility: loaded ? 'visible' : 'hidden',
        }}
      >
        <img
          ref={imgRef}
          src={src}
          alt={alt}
          draggable={false}
          style={{ display: 'block', maxWidth: 'none' }}
          onLoad={() => {
            setLoaded(true)
            fitToContainer()
          }}
        />

        {transformOverlay && (
          <div style={{ position: 'absolute', inset: 0 }}>
            {transformOverlay}
          </div>
        )}
      </div>

      {/* ── Zoom controls ────────────────────────────────────────────────── */}
      <div className="absolute top-2 right-2 z-10 flex gap-1">
        <button
          onPointerDown={btnDown}
          onClick={btnClick(() => zoomCenter(ZOOM_STEP ** 3))}
          className="bg-white/90 hover:bg-white p-1.5 rounded shadow-sm"
          title="Zoom in"
        >
          <ZoomIn size={16} />
        </button>
        <button
          onPointerDown={btnDown}
          onClick={btnClick(() => zoomCenter(1 / ZOOM_STEP ** 3))}
          className="bg-white/90 hover:bg-white p-1.5 rounded shadow-sm"
          title="Zoom out"
        >
          <ZoomOut size={16} />
        </button>
        <button
          onPointerDown={btnDown}
          onClick={btnClick(fitToContainer)}
          className="bg-white/90 hover:bg-white p-1.5 rounded shadow-sm"
          title="Fit to screen"
        >
          <Maximize size={16} />
        </button>
      </div>

      {/* ── Static overlay (non-transformed, pointer-events-none) ─────────── */}
      {overlay && (
        <div className="absolute inset-0 z-10 pointer-events-none">
          {overlay}
        </div>
      )}
    </div>
  )
}
