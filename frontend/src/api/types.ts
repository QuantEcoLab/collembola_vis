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
  image_id?: string
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

export interface JobResult {
  csv_path?: string
  excel_path?: string
  overlay_path?: string
  [key: string]: unknown
}

export interface Job {
  id: string
  type: 'detection' | 'measurement' | 'calibration' | 'batch' | 'batch_measure' | 'batch_process' | 'finetune' | 'contours'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message: string
  result: JobResult
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
  folder: string | null
}

export interface ProjectDetail extends Project {
  images: ProjectImage[]
}

export interface MeasurementRequest {
  image_id: string
  detections_csv: string
  um_per_pixel: number
  method?: 'centerline_circular_sections_v2'
  device?: string
  use_annotations?: boolean
}

export interface ContourRequest {
  image_id: string
  detection_job_id?: string
  source_contour_job_id?: string
  um_per_pixel: number | null
  use_annotations?: boolean
}

export const CENTERLINE_METHOD = 'centerline_circular_sections_v2'

export interface VolumeSummary {
  available: boolean
  count: number
  unit: string | null
  mean: number | null
  median: number | null
  min: number | null
  max: number | null
  total: number | null
  reliable_count?: number
  excluded_count?: number
  candidate_count?: number
  uncalibrated_count?: number
}

export interface OrganismContour {
  detection_id: number
  bbox: [number, number, number, number]
  contours: [number, number][][]
  centerline?: [number, number][]
  cross_sections?: { left: [number, number] | null; right: [number, number] | null }[]
  removed_branches?: [number, number][][]
  measurement_reliable?: boolean
}

export interface DetectorDatasetEntry {
  entry_id: string; revision: number; image_id: string; filename: string
  image_sha256: string; width: number; height: number; saved_at: string; saved_by: string
  source: { description: string; resolved_path: string; ingestion: string; exif_date_original: string | null }
  boxes: (AnnotatedBox & { action: 'added' | 'corrected' | 'confirmed'; original_bbox: Record<string, number> | null })[]
  removed_detections: { id: string }[]
  change_counts: { added: number; corrected: number; confirmed: number }
  split_check: { membership: string; registry_created_at: string }
}
export interface DetectorDatasetVersion {
  version_id: string; parent_version: string | null; created_at: string; created_by: string
  entries: DetectorDatasetEntry[]
}
export interface DetectorDatasetIndex {
  versions: (Omit<DetectorDatasetVersion, 'entries'> & { image_count: number; box_count: number })[]
  registry: { created_at: string; roots: string[]; sources: number; images: number } | null
}
