import { useState } from 'react'

interface Point {
  x: number
  y: number
}

interface Props {
  /** Called with two points in *original image* coordinates */
  onPointsSelected: (p1: [number, number], p2: [number, number]) => void
  /** Scale factor: original pixels / displayed pixels */
  scale: number
}

export default function CalibrationTool({ onPointsSelected, scale }: Props) {
  const [points, setPoints] = useState<Point[]>([])

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const displayX = e.clientX - rect.left
    const displayY = e.clientY - rect.top
    const origX = displayX * scale
    const origY = displayY * scale

    const next = [...points, { x: origX, y: origY }]
    if (next.length >= 2) {
      onPointsSelected([next[0].x, next[0].y], [next[1].x, next[1].y])
      setPoints([])
    } else {
      setPoints(next)
    }
  }

  return (
    <div
      className="absolute inset-0 cursor-crosshair"
      onClick={handleClick}
    >
      {points.map((p, i) => (
        <div
          key={i}
          className="absolute w-3 h-3 -translate-x-1/2 -translate-y-1/2 bg-red-500 rounded-full border-2 border-white shadow"
          style={{ left: p.x / scale, top: p.y / scale }}
        />
      ))}
      {points.length === 1 && (
        <p className="absolute bottom-2 left-2 text-xs bg-black/60 text-white px-2 py-1 rounded">
          Click the second point on the ruler
        </p>
      )}
      {points.length === 0 && (
        <p className="absolute bottom-2 left-2 text-xs bg-black/60 text-white px-2 py-1 rounded">
          Click the first point on the ruler
        </p>
      )}
    </div>
  )
}
