import type {
  CalibrationResult,
  DetectionRequest,
  ImageInfo,
  Job,
  MeasurementRequest,
} from './types'

const BASE = ''  // proxied by Vite in dev

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

// Images
export const uploadImage = async (file: File): Promise<ImageInfo> => {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/api/images/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('Upload failed')
  return res.json()
}

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

// Thumbnail URL helper
export const thumbnailUrl = (imageId: string) =>
  `${BASE}/files/uploads/${imageId}/thumbnail.jpg`

// Output file URL helper
export const outputFileUrl = (jobId: string, filename: string) =>
  `${BASE}/files/outputs/${jobId}/${filename}`
