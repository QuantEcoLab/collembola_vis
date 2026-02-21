export interface ImageInfo {
  image_id: string
  filename: string
  path: string
  thumbnail_path: string | null
  width: number
  height: number
  thumbnail_width?: number
  thumbnail_height?: number
}

export interface CalibrationResult {
  um_per_pixel: number | null
  ruler_px?: number
  tick_spacing_px?: number
  num_ticks?: number
  confidence: number
  method: string
  calibration_id?: string
  error?: string
  point1?: number[]
  point2?: number[]
  known_mm?: number
}

export interface Job {
  id: string
  type: 'detection' | 'measurement' | 'calibration' | 'batch'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message: string
  result: Record<string, any>
  error: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface DetectionRequest {
  image_id: string
  model_path?: string
  conf?: number
  iou?: number
  tile_size?: number
  overlap?: number
  device?: string
}

export interface MeasurementRequest {
  image_id: string
  detections_csv: string
  um_per_pixel: number
  method?: 'fast' | 'sam'
  device?: string
}
