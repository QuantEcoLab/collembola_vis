import type {
  AnnotatedBox,
  AnnotationFile,
  CalibrationResult,
  DetectionRequest,
  ImageInfo,
  Job,
  MeasurementRequest,
  ModelInfo,
  Project,
  ProjectDetail,
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
  window.location.href = `${BASE}/login`
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
  `${BASE}/files/outputs/${jobId}/${encodeURIComponent(filename)}`

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

// Projects
export const listProjects = () => request<Project[]>('/api/projects')

export const getProject = (id: string) => request<ProjectDetail>(`/api/projects/${id}`)

export const createProject = (name: string, description: string) =>
  request<{ id: string; name: string }>('/api/projects', {
    method: 'POST',
    body: JSON.stringify({ name, description }),
  })

export const updateProject = (id: string, name: string, description: string) =>
  request<{ ok: boolean }>(`/api/projects/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ name, description }),
  })

export const deleteProject = (id: string) =>
  request<{ ok: boolean }>(`/api/projects/${id}`, { method: 'DELETE' })

export const addImageToProject = (projectId: string, imageId: string, filename: string) =>
  request<{ id: string }>(`/api/projects/${projectId}/images`, {
    method: 'POST',
    body: JSON.stringify({ image_id: imageId, filename }),
  })

export const removeImageFromProject = (projectId: string, imageId: string) =>
  request<{ ok: boolean }>(`/api/projects/${projectId}/images/${imageId}`, { method: 'DELETE' })

export const runBatchDetection = (
  projectId: string,
  params: { conf?: number; tile_size?: number; overlap?: number; device?: string },
) =>
  request<{ job_id: string; status: string }>(`/api/projects/${projectId}/detect`, {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const runBatchMeasurement = (
  projectId: string,
  params: { um_per_pixel: number; method?: string; device?: string },
) =>
  request<{ job_id: string; status: string }>(`/api/projects/${projectId}/measure`, {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const runBatchProcess = (
  projectId: string,
  params: { um_per_pixel: number; conf?: number; device?: string },
) =>
  request<{ job_id: string; status: string }>(`/api/projects/${projectId}/process`, {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const updateProjectImageJobs = (
  projectId: string,
  imageId: string,
  jobs: { detection_job_id?: string; measurement_job_id?: string },
) =>
  request<{ ok: boolean }>(`/api/projects/${projectId}/images/${imageId}/jobs`, {
    method: 'PATCH',
    body: JSON.stringify(jobs),
  })

export const projectAnnotationsExportUrl = (projectId: string, format: 'json' | 'csv') =>
  `${BASE}/api/projects/${projectId}/annotations/export?format=${format}`

export const listProjectFolders = (projectId: string) =>
  request<string[]>(`/api/projects/${projectId}/folders`)

export const setImageFolder = (projectId: string, imageId: string, folder: string | null) =>
  request<{ ok: boolean }>(`/api/projects/${projectId}/images/${imageId}/folder`, {
    method: 'PATCH',
    body: JSON.stringify({ folder }),
  })

export const deleteProjectFolder = (projectId: string, folderName: string) =>
  request<{ ok: boolean }>(`/api/projects/${projectId}/folders/${encodeURIComponent(folderName)}`, {
    method: 'DELETE',
  })

export const renameProjectFolder = (projectId: string, oldName: string, newName: string) =>
  request<{ ok: boolean }>(`/api/projects/${projectId}/folders/${encodeURIComponent(oldName)}`, {
    method: 'PATCH',
    body: JSON.stringify({ new_name: newName }),
  })

/** Fetch all measurement CSVs as a ZIP and trigger a browser download. */
export async function downloadProjectResults(projectId: string, projectName: string): Promise<void> {
  const res = await fetch(`${BASE}/api/projects/${projectId}/results/download`, {
    headers: { ...authHeaders() },
  })
  if (res.status === 401) handle401()
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `HTTP ${res.status}`)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const safe = projectName.replace(/[^a-zA-Z0-9-_]/g, '_')
  a.download = `${safe}_measurements.zip`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
