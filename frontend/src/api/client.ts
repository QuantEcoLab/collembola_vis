import type {
  AnnotatedBox,
  AnnotationFile,
  CalibrationResult,
  CommunityEntry,
  CommunityStats,
  DetectionRequest,
  ImageInfo,
  Job,
  MeasurementRequest,
  ModelInfo,
} from './types'

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '')  // Use Vite's BASE_URL, remove trailing slash

/** Read the JWT from the Zustand persist store in localStorage. */
function getToken(): string | null {
  try {
    const raw = localStorage.getItem('auth')
    if (!raw) return null
    return (JSON.parse(raw) as { state?: { token?: string } }).state?.token ?? null
  } catch {
    return null
  }
}

function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function handle401(): never {
  localStorage.removeItem('auth')
  window.location.href = '/login'
  throw new Error('Unauthorized')
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    ...options,
  })
  if (!res.ok) {
    if (res.status === 401) handle401()
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

// Images
export const uploadImage = (
  file: File,
  onProgress?: (pct: number) => void,
): Promise<ImageInfo> => {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/api/images/upload`)

    const token = getToken()
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status === 401) { handle401(); return }
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)) }
        catch { reject(new Error('Invalid server response')) }
      } else {
        let detail = `Upload failed (HTTP ${xhr.status})`
        try { detail = JSON.parse(xhr.responseText).detail || detail } catch {}
        reject(new Error(detail))
      }
    }

    xhr.onerror = () => reject(new Error('Network error — check your connection'))
    xhr.ontimeout = () => reject(new Error('Upload timed out'))

    xhr.send(form)
  })
}

export const registerFromPath = (path: string) =>
  request<ImageInfo>('/api/images/from-path', {
    method: 'POST',
    body: JSON.stringify({ path }),
  })

export const listImages = () => request<ImageInfo[]>('/api/images')

export const getImage = (id: string) => request<ImageInfo>(`/api/images/${id}`)

export const deleteImage = (id: string) =>
  request<{ ok: boolean }>(`/api/images/${id}`, { method: 'DELETE' })

// Calibration
export const autoCalibrate = (image_id: string, ruler_mm = 10) =>
  request<CalibrationResult>('/api/calibration/auto', {
    method: 'POST',
    body: JSON.stringify({ image_id, ruler_mm }),
  })

export const manualCalibrate = (
  image_id: string,
  point1: [number, number],
  point2: [number, number],
  known_mm: number,
) =>
  request<CalibrationResult>('/api/calibration/manual', {
    method: 'POST',
    body: JSON.stringify({ image_id, point1, point2, known_mm }),
  })

export const getCalibration = (image_id: string) =>
  request<CalibrationResult>(`/api/calibration/${image_id}`)

// Detection
export const runDetection = (req: DetectionRequest) =>
  request<{ job_id: string; status: string }>('/api/detection/run', {
    method: 'POST',
    body: JSON.stringify(req),
  })

export const getDetectionResult = (jobId: string) =>
  request<Job>(`/api/detection/result/${jobId}`)

// Measurement
export const runMeasurement = (req: MeasurementRequest) =>
  request<{ job_id: string; status: string }>('/api/measurement/run', {
    method: 'POST',
    body: JSON.stringify(req),
  })

export const getMeasurementResult = (jobId: string) =>
  request<Job>(`/api/measurement/result/${jobId}`)

// Jobs
export const listJobs = () => request<Job[]>('/api/jobs')
export const getJob = (id: string) => request<Job>(`/api/jobs/${id}`)

// Full image URL (original file as uploaded / symlinked)
export const imageUrl = (imageId: string, filename: string) =>
  `${BASE}/files/uploads/${imageId}/${encodeURIComponent(filename)}`

// Thumbnail URL helper (kept for compatibility)
export const thumbnailUrl = (imageId: string) =>
  `${BASE}/files/uploads/${imageId}/thumbnail.jpg`

// Output file URL helper
export const outputFileUrl = (jobId: string, filename: string) =>
  `${BASE}/files/outputs/${jobId}/${filename}`

// Detection boxes (raw CSV as JSON)
export const getDetectionBoxes = (jobId: string) =>
  request<{ boxes: AnnotatedBox[] }>(`/api/detection/boxes/${jobId}`)

// Annotations
export const getAnnotations = (imageId: string) =>
  request<AnnotationFile>(`/api/annotations/${imageId}`)

export const saveAnnotations = (imageId: string, data: AnnotationFile) =>
  request<{ ok: boolean }>(`/api/annotations/${imageId}`, {
    method: 'POST',
    body: JSON.stringify(data),
  })

export const annotationExportUrl = (imageId: string, fmt: 'json' | 'csv') =>
  `${BASE}/api/annotations/${imageId}/export?format=${fmt}`

// Models
export const listModels = () => request<ModelInfo[]>('/api/models')

// Fine-tune
export interface FinetuneRequest {
  image_id: string
  annotation_file: string
  base_model?: string
  epochs?: number
  device?: string
  tile_size?: number
  overlap?: number
}

export const runFinetune = (req: FinetuneRequest) =>
  request<{ job_id: string; status: string }>('/api/finetune/run', {
    method: 'POST',
    body: JSON.stringify(req),
  })

// Community
export interface CommunitySubmitRequest {
  image_name: string
  image_width?: number | null
  image_height?: number | null
  um_per_pixel?: number | null
  conf_threshold?: number | null
  boxes: AnnotatedBox[]
}

export const submitToCommunity = (req: CommunitySubmitRequest) =>
  request<{ id: string; num_detections: number }>('/api/community/submit', {
    method: 'POST',
    body: JSON.stringify(req),
  })

export const listCommunity = (limit = 20, offset = 0, search = '') => {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (search) params.set('search', search)
  return request<CommunityEntry[]>(`/api/community/list?${params}`)
}

export const getCommunityEntry = (id: string) =>
  request<CommunityEntry & { boxes: Record<string, any>[] }>(`/api/community/${id}`)

export const getCommunityStats = () =>
  request<CommunityStats>('/api/community/stats')

export const communityExportUrl = (id: string, format: 'json' | 'csv') =>
  `${BASE}/api/community/${id}/export?format=${format}`
