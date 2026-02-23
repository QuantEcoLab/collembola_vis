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
  type: 'detection' | 'measurement' | 'calibration' | 'batch' | 'finetune'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message: string
  result: Record<string, any>
  error: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface AnnotatedBox {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  status: 'accepted' | 'rejected' | 'added'
}

export interface AnnotationFile {
  image_id: string
  image_filename: string
  source_job_id: string
  created_at: string
  boxes: AnnotatedBox[]
}

export interface CommunityEntry {
  id: string
  username: string
  submitted_at: string
  image_name: string
  image_width: number | null
  image_height: number | null
  num_detections: number
  um_per_pixel: number | null
  conf_threshold: number | null
}

export interface CommunityStats {
  total_submissions: number
  total_detections: number
  total_users: number
  total_images: number
}

export interface ModelInfo {
  name: string
  path: string
  size_mb: number
  mtime: string
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
  use_annotations?: boolean
}
