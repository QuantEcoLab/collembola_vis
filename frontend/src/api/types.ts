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

export interface Project {
  id: string
  name: string
  description: string
  created_by: string
  created_at: string
  updated_at: string
  image_count: number
  thumbnail_image_id: string | null
}

export interface ProjectImage {
  id: string
  project_id: string
  image_id: string
  filename: string
  added_by: string
  added_at: string
  detection_job_id: string | null
  measurement_job_id: string | null
  has_annotation: boolean
  annotation_total: number
  annotation_accepted: number
}

export interface ProjectDetail extends Project {
  images: ProjectImage[]
}

export interface MeasurementRequest {
  image_id: string
  detections_csv: string
  um_per_pixel: number
  method?: 'fast' | 'sam'
  device?: string
  use_annotations?: boolean
}
