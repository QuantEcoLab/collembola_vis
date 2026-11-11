import type { OrganismContour } from '../api/types'

export default function ContourHighlight({ contour, width, height }: { contour: OrganismContour | null; width: number; height: number }) {
  if (!contour) return null
  return (
    <svg data-testid="selected-contour" data-detection-id={contour.detection_id} viewBox={`0 0 ${width} ${height}`}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
      {contour.contours.map((points, index) => <polygon key={index} points={points.map(p => p.join(',')).join(' ')}
        fill="rgba(249,115,22,0.18)" stroke="#ff4800" strokeWidth={0.6} vectorEffect="non-scaling-stroke" />)}
    </svg>
  )
}
