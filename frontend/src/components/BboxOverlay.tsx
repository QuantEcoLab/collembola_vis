import { useRef, useMemo } from 'react'
import type { AnnotatedBox } from '../api/types'

interface DrawingBox {
  x1: number; y1: number; x2: number; y2: number
}

interface Props {
  boxes: AnnotatedBox[]
  imageWidth: number
  imageHeight: number
  onBoxClick?: (id: string) => void
  drawingBox?: DrawingBox | null
  mode: 'review' | 'draw'
  selectedId?: string | null
  onDrawStart?: (x: number, y: number) => void
  onDrawMove?: (x: number, y: number) => void
  onDrawEnd?: (x: number, y: number) => void
}

const statusStroke: Record<AnnotatedBox['status'], string> = {
  accepted: '#22c55e',
  rejected: '#ef4444',
  added: '#3b82f6',
}

const statusFill: Record<AnnotatedBox['status'], string> = {
  accepted: 'rgba(34,197,94,0.10)',
  rejected: 'rgba(239,68,68,0.15)',
  added: 'rgba(59,130,246,0.10)',
}

export default function BboxOverlay({
  boxes,
  imageWidth,
  imageHeight,
  onBoxClick,
  drawingBox,
  mode,
  selectedId,
  onDrawStart,
  onDrawMove,
  onDrawEnd,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)

  function toImageCoords(e: React.MouseEvent): { x: number; y: number } {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const rect = svg.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * imageWidth
    const y = ((e.clientY - rect.top) / rect.height) * imageHeight
    return { x, y }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (mode !== 'draw' || !onDrawStart) return
    e.preventDefault()
    const { x, y } = toImageCoords(e)
    onDrawStart(x, y)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (mode !== 'draw' || !onDrawMove) return
    const { x, y } = toImageCoords(e)
    onDrawMove(x, y)
  }

  const handleMouseUp = (e: React.MouseEvent) => {
    if (mode !== 'draw' || !onDrawEnd) return
    const { x, y } = toImageCoords(e)
    onDrawEnd(x, y)
  }

  // Flag non-rejected boxes whose IoU with any other non-rejected box is >= 0.5
  const overlappingIds = useMemo(() => {
    const active = boxes.filter((b) => b.status !== 'rejected')
    const ids = new Set<string>()
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const a = active[i], b = active[j]
        const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1))
        const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1))
        const inter = ix * iy
        if (inter === 0) continue
        const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        const overlapRatio = inter / Math.min(areaA, areaB)
        if (overlapRatio >= 0.5) {
          ids.add(a.id)
          ids.add(b.id)
        }
      }
    }
    return ids
  }, [boxes])

  const sw = Math.max(2, Math.round(imageWidth / 1000))

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      style={{
        width: '100%',
        height: '100%',
        position: 'absolute',
        inset: 0,
        cursor: mode === 'draw' ? 'crosshair' : 'default',
        userSelect: 'none',
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {boxes.map((box) => {
        const isSelected = box.id === selectedId
        const isRejected = box.status === 'rejected'
        const isOverlapping = !isSelected && !isRejected && overlappingIds.has(box.id)
        const stroke = isSelected ? '#f59e0b' : isOverlapping ? '#f97316' : statusStroke[box.status]
        const fill = isSelected
          ? 'rgba(245,158,11,0.10)'
          : isOverlapping
          ? 'rgba(249,115,22,0.10)'
          : statusFill[box.status]
        const strokeWidth = isSelected ? sw * 1.5 : isRejected ? sw * 1.2 : sw
        const strokeOpacity = isSelected ? 0.85 : isRejected ? 0.70 : 0.35

        return (
          <g key={box.id}>
            {/* Box body */}
            <rect
              x={box.x1}
              y={box.y1}
              width={box.x2 - box.x1}
              height={box.y2 - box.y1}
              stroke={stroke}
              strokeWidth={strokeWidth}
              strokeOpacity={strokeOpacity}
              strokeDasharray={box.status === 'added' && !isSelected ? `${sw * 4} ${sw * 2}` : undefined}
              fill={fill}
              style={{
                cursor: mode === 'review' ? 'pointer' : 'default',
                pointerEvents: mode === 'review' ? 'auto' : 'none',
              }}
              onPointerDown={mode === 'review' ? (e) => e.stopPropagation() : undefined}
              onClick={mode === 'review' ? () => onBoxClick?.(box.id) : undefined}
            />

            {/* Rejected cross-out line */}
            {isRejected && !isSelected && (
              <line
                x1={box.x1}
                y1={box.y1}
                x2={box.x2}
                y2={box.y2}
                stroke="#ef4444"
                strokeWidth={sw * 1.2}
                opacity={0.55}
                pointerEvents="none"
              />
            )}

          </g>
        )
      })}

      {/* In-progress draw rectangle */}
      {drawingBox && (
        <rect
          x={Math.min(drawingBox.x1, drawingBox.x2)}
          y={Math.min(drawingBox.y1, drawingBox.y2)}
          width={Math.abs(drawingBox.x2 - drawingBox.x1)}
          height={Math.abs(drawingBox.y2 - drawingBox.y1)}
          stroke="#94a3b8"
          strokeWidth={sw}
          strokeOpacity={0.5}
          strokeDasharray={`${sw * 4} ${sw * 2}`}
          fill="rgba(148,163,184,0.08)"
          pointerEvents="none"
        />
      )}
    </svg>
  )
}
