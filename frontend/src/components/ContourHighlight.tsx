import type { OrganismContour } from '../api/types'

export default function ContourHighlight({ contour, width, height }: { contour: OrganismContour | null; width: number; height: number }) {
  if (!contour) return null
  return (
    <svg data-testid="selected-contour" data-detection-id={contour.detection_id} viewBox={`0 0 ${width} ${height}`}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
      {contour.contours.map((points, index) => <polygon key={index} points={points.map(p => p.join(',')).join(' ')}
         fill="rgba(249,115,22,0.18)" stroke="#ff4800" strokeWidth={0.6} vectorEffect="non-scaling-stroke" />)}
      {contour.cross_sections?.map((section, i) => section.left && section.right && <line key={`width-${i}`}
        x1={section.left[0]} y1={section.left[1]} x2={section.right[0]} y2={section.right[1]}
        stroke="#00d4ff" strokeWidth={1} vectorEffect="non-scaling-stroke" />)}
      {contour.removed_branches?.map((points, i) => <polyline key={`removed-${i}`} points={points.map(p => p.join(',')).join(' ')}
        fill="none" stroke="#ff32ca" strokeWidth={1} vectorEffect="non-scaling-stroke" />)}
      {contour.centerline && <polyline data-testid="selected-centerline" points={contour.centerline.map(p => p.join(',')).join(' ')}
        fill="none" stroke={contour.measurement_reliable === false ? '#ff9900' : '#45ff60'} strokeWidth={1.7} vectorEffect="non-scaling-stroke" />}
    </svg>
  )
}
