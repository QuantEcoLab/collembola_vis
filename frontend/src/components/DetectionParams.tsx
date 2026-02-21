interface Params {
  conf: number
  iou: number
  tileSize: number
  overlap: number
}

interface Props {
  params: Params
  onChange: (params: Params) => void
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-600">{label}</span>
        <span className="font-mono text-gray-900">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  )
}

export default function DetectionParams({ params, onChange }: Props) {
  return (
    <div className="space-y-4">
      <Slider
        label="Confidence threshold"
        value={params.conf}
        min={0.1}
        max={0.95}
        step={0.05}
        onChange={(conf) => onChange({ ...params, conf })}
      />
      <Slider
        label="IoU threshold (NMS)"
        value={params.iou}
        min={0.1}
        max={0.9}
        step={0.05}
        onChange={(iou) => onChange({ ...params, iou })}
      />
      <Slider
        label="Tile size"
        value={params.tileSize}
        min={640}
        max={2560}
        step={128}
        onChange={(tileSize) => onChange({ ...params, tileSize })}
      />
      <Slider
        label="Tile overlap"
        value={params.overlap}
        min={64}
        max={512}
        step={64}
        onChange={(overlap) => onChange({ ...params, overlap })}
      />
    </div>
  )
}

export type { Params as DetectionParamsType }
