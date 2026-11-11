import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogOut, ChevronLeft, ChevronRight, Table2, MousePointer2, Square, XCircle, Pencil, CheckCircle2 } from 'lucide-react'
import ImageUploader from '../components/ImageUploader'
import ImageViewer from '../components/ImageViewer'
import BboxOverlay from '../components/BboxOverlay'
import { AdvancedDetectionModal, STANDARD_CONF } from '../components/AdvancedDetectionModal'
import { ViewerToolbar, type OverlayMode } from '../components/ViewerToolbar'
import { MeasurementModal } from '../components/MeasurementModal'
import { WorkspaceSidebar } from '../components/WorkspaceSidebar'
import ExperimentalContourAction from '../components/ExperimentalContourAction'
import ContourHighlight from '../components/ContourHighlight'
import SaveDetectorDataset from '../components/SaveDetectorDataset'
import BoxCoordinateEditor from '../components/BoxCoordinateEditor'
import {
  runDetection,
  manualCalibrate,
  getCalibration,
  imageUrl,
  outputFileUrl,
  updateProjectImageJobs,
  getProject,
  getImage,
  getJob,
  runContours,
  getContourSource,
  getContourOverview,
} from '../api/client'
import { useJobProgress } from '../hooks/useJob'
import { useRefinement } from '../hooks/useRefinement'
import { useCalibrationStore } from '../store/calibrationStore'
import { useAuthStore } from '../store/authStore'
import { useWorkspaceStore } from '../store/workspaceStore'
import { useProjectStore } from '../store/projectStore'
import type { ImageInfo, ProjectImage, OrganismContour, VolumeSummary } from '../api/types'
import { CENTERLINE_METHOD } from '../api/types'

type ModalType = 'advanced-detection' | null
type Point = [number, number]

const basename = (path: string) => path.split(/[\\/]/).pop()

export default function WorkspacePage() {
  const navigate = useNavigate()
  const logout = useAuthStore((s) => s.logout)
  const role = useAuthStore((s) => s.role)
  const calStore = useCalibrationStore()
  const workspaceStore = useWorkspaceStore()
  const { currentProjectId, currentProjectName } = useProjectStore()

  // ── Core State ──────────────────────────────────────────────────────
  const [image, setImageState] = useState<ImageInfo | null>(workspaceStore.image)
  const setImage = (img: ImageInfo | null) => {
    setImageState(img)
    workspaceStore.setImage(img)
    setContourJobIdState(null)
    workspaceStore.setContourJobId(null)
    setContourError(null)
    setOrganismContours([])
    setSelectedContourId(null)
  }

  // ── Workflow State ──────────────────────────────────────────────────
  const [modalOpen, setModalOpen] = useState<ModalType>(null)

  // ── Calibration ─────────────────────────────────────────────────────
  const [umPerPixel, setUmPerPixelState] = useState<number>(calStore.umPerPixel ?? 8.57)
  const setUmPerPixel = (v: number) => {
    const valid = Number.isFinite(v) && v > 0
    setUmPerPixelState(Number.isFinite(v) ? v : 0)
    if (valid) calStore.setUmManual(v)
    if (image) calStore.setImageCalibration(image.image_id, valid ? v : null)
  }
  const [calibrationMode, setCalibrationMode] = useState(false)
  const [calibrationPoints, setCalibrationPoints] = useState<Point[]>([])
  const [calibrationError, setCalibrationError] = useState<string | null>(null)
  const [calibrationMessage, setCalibrationMessage] = useState<string | null>(null)
  // ── Detection ───────────────────────────────────────────────────────
  // Initial value of the advanced dialog only; the standard run sends no thresholds so the backend's
  // detection profile (fine-tuned detector on 1280 tiles, fragment suppression) applies.
  const [conf] = useState(STANDARD_CONF)
  const [tileSize] = useState(1280)
  const [overlap] = useState(256)
  const [device] = useState('0')
  const [detectionJobId, setDetectionJobIdState] = useState<string | null>(
    workspaceStore.detectionJobId
  )
  const setDetectionJobId = (id: string | null) => {
    setDetectionJobIdState(id)
    workspaceStore.setDetectionJobId(id)
  }
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const detectionJob = useJobProgress(detectionJobId)
  const detectionDone = detectionJob?.status === 'completed'

  // ── Refinement (Annotation Editing) ─────────────────────────────────
  type EditTool = 'select' | 'reject' | 'draw'
  const [refineMode, setRefineMode] = useState(false)
  const [editTool, setEditTool] = useState<EditTool>('select')
  const [refineSaveError, setRefineSaveError] = useState<string | null>(null)
  const refinement = useRefinement(
    image?.image_id ?? null,
    detectionJobId,
    detectionDone,
  )

  // ── Measurement ─────────────────────────────────────────────────────
  const [measureJobId, setMeasureJobIdState] = useState<string | null>(
    workspaceStore.measureJobId
  )
  const setMeasureJobId = (id: string | null) => {
    setMeasureJobIdState(id)
    workspaceStore.setMeasureJobId(id)
  }
  const [measureError, setMeasureError] = useState<string | null>(null)
  const measureJob = useJobProgress(measureJobId)
  const [csvData, setCsvData] = useState<Record<string, any>[] | null>(null)
  const measurementDone = measureJob?.status === 'completed' && measureJob.result?.measurement_method === CENTERLINE_METHOD && measureJob.result.image_id === image?.image_id

  // ── Fixed experimental contours + automatic existing mask measurement ──
  const [contourJobId, setContourJobIdState] = useState<string | null>(workspaceStore.contourJobId ?? null)
  const [contourSubmitting, setContourSubmitting] = useState(false)
  const contourSubmitLock = useRef(false)
  const activeImageRef = useRef(image?.image_id)
  activeImageRef.current = image?.image_id
  const [contourError, setContourError] = useState<string | null>(null)
  const [availableContourSource, setAvailableContourSource] = useState<string | null>(null)
  const [organismContours, setOrganismContours] = useState<OrganismContour[]>([])
  const [selectedContourId, setSelectedContourId] = useState<number | null>(null)
  const contourJob = useJobProgress(contourJobId)
  const contourBusy = contourSubmitting || (contourJobId != null && (!contourJob || contourJob.status === 'pending' || contourJob.status === 'running'))
  const contourCalibration = image ? calStore.imageUmPerPixel?.[image.image_id] ?? null : null
  const isExperimentalMeasurement = measurementDone
  const needsCurrentMeasurement = measureJob?.status === 'completed' && !measurementDone
  const calibrationMatchesResult = !isExperimentalMeasurement || (measureJob?.result.calibration_um_per_px ?? null) === contourCalibration
  const volumeSummary = isExperimentalMeasurement && calibrationMatchesResult ? measureJob?.result.volume_summary as VolumeSummary : null
  const selectedContour = useMemo(() => organismContours.find(c => c.detection_id === selectedContourId) ?? null, [organismContours, selectedContourId])
  const selectedContourBox = useMemo(() => selectedContour ? { x1: selectedContour.bbox[0], y1: selectedContour.bbox[1], x2: selectedContour.bbox[2], y2: selectedContour.bbox[3] } : null, [selectedContour])

  const submitContours = useCallback(async (sourceContourJobId?: string) => {
    if (!image) { setContourError('Najprije učitajte fotografiju.'); return }
    if (contourSubmitLock.current || contourBusy) return
    if (refinement.hasUnsavedEdits) {
      // Measuring uses the saved annotation file or the detection job, never unsaved edits (e.g. a merge).
      setContourError('Spremite izmjene pravokutnika (Save) prije mjerenja; inače se ne bi mjerili prikazani pravokutnici.')
      return
    }
    const imageId = image.image_id
    contourSubmitLock.current = true
    setContourSubmitting(true)
    setContourError(null)
    try {
      const cachedSource = sourceContourJobId ?? availableContourSource ?? (measureJob?.result?.model_name === 'exp03_no_os' && measureJob.result.image_id === imageId && measureJob.result.mask_records_path ? measureJob.id : undefined)
      const useAnnotations = refinement.annotationsSaved
      const currentDetections = detectionJobId && detectionDone ? detectionJobId : undefined
      const response = await runContours({ image_id: imageId, um_per_pixel: contourCalibration,
        ...(useAnnotations ? { use_annotations: true } : {}),
        // Without saved annotations the shown boxes are the current detection job's; the backend reuses
        // cached masks only if they were made from that job, otherwise it segments the job's boxes.
        ...(cachedSource && (useAnnotations || !currentDetections)
          ? { source_contour_job_id: cachedSource }
          : currentDetections ? { detection_job_id: currentDetections, ...(cachedSource ? { source_contour_job_id: cachedSource } : {}) } : {}),
      })
      if (activeImageRef.current !== imageId) return
      setContourJobIdState(response.job_id)
      workspaceStore.setContourJobId(response.job_id)
    } catch (error) {
      if (activeImageRef.current === imageId) setContourError(error instanceof Error ? error.message : 'Izrada kontura nije uspjela.')
    } finally {
      contourSubmitLock.current = false
      setContourSubmitting(false)
    }
  }, [image, contourBusy, contourCalibration, detectionJobId, detectionDone, workspaceStore.setContourJobId, availableContourSource, refinement.annotationsSaved, refinement.hasUnsavedEdits, measureJob?.id, measureJob?.result])

  useEffect(() => {
    let cancelled = false
    setAvailableContourSource(null)
    if (image) getContourSource(image.image_id).then(data => { if (!cancelled) setAvailableContourSource(data.job_id) }).catch(() => {})
    return () => { cancelled = true }
  }, [image?.image_id])

  useEffect(() => {
    if (!contourJobId) return
    let cancelled = false
    getJob(contourJobId).catch(error => {
      if (!cancelled) {
        setContourError(`Posao kontura nije dostupan: ${error.message}`)
        setContourJobIdState(null)
        workspaceStore.setContourJobId(null)
      }
    })
    return () => { cancelled = true }
  }, [contourJobId, workspaceStore.setContourJobId])

  useEffect(() => {
    if (contourJob?.status === 'failed') setContourError(`Izrada kontura i mjerenja nije uspjela: ${contourJob.error ?? 'nepoznata greška'}`)
    if (contourJob?.status !== 'completed' || contourJob.result.image_id !== image?.image_id || contourJob.result.measurement_method !== CENTERLINE_METHOD) return
    setAvailableContourSource(contourJob.id)
    if (measureJobId !== contourJob.id) {
      setMeasureJobId(contourJob.id)
      setOverlayMode('contours')
    }
    if (!detectionJobId && typeof contourJob.result.detection_job_id === 'string') setDetectionJobId(contourJob.result.detection_job_id)
  }, [contourJob?.status, contourJob?.id, contourJob?.result, image?.image_id, measureJobId, detectionJobId])

  useEffect(() => {
    if (!isExperimentalMeasurement || contourBusy || contourJob?.status === 'failed' || !measureJob) return
    if ((measureJob.result.calibration_um_per_px ?? null) === contourCalibration) return
    const sourceId = measureJob.id
    const timer = setTimeout(() => { void submitContours(sourceId) }, 650)
    return () => clearTimeout(timer)
  }, [isExperimentalMeasurement, contourBusy, contourJob?.status, measureJob?.id, measureJob?.result.calibration_um_per_px, contourCalibration, submitContours])

  useEffect(() => {
    if (!isExperimentalMeasurement || !measureJob || typeof measureJob.result.contours_path !== 'string') {
      setOrganismContours([])
      return
    }
    let cancelled = false
    const filename = basename(measureJob.result.contours_path)!
    fetch(outputFileUrl(measureJob.id, filename)).then(response => {
      if (!response.ok) throw new Error('Konture nisu dostupne za prikaz.')
      return response.json()
    }).then(data => { if (!cancelled) setOrganismContours(data.organisms) }).catch(error => { if (!cancelled) setContourError(error.message) })
    return () => { cancelled = true }
  }, [isExperimentalMeasurement, measureJob?.id, measureJob?.result.contours_path])

  // ── View State ──────────────────────────────────────────────────────
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('raw')
  const prevOverlayModeRef = useRef<OverlayMode>('boxes')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [showMeasurementModal, setShowMeasurementModal] = useState(false)

  // ── Project image list (for next/prev navigation) ──────────────────
  const [projectImages, setProjectImages] = useState<ProjectImage[]>([])

  useEffect(() => {
    if (!currentProjectId) { setProjectImages([]); return }
    getProject(currentProjectId)
      .then((p) => setProjectImages(p.images))
      .catch(() => {})
  }, [currentProjectId])

  useEffect(() => {
    if (!image) return
    const storedImageScale = calStore.imageUmPerPixel?.[image.image_id]
    if (storedImageScale != null && Number.isFinite(storedImageScale) && storedImageScale > 0) {
      setUmPerPixelState(storedImageScale)
      return
    }
    let cancelled = false
    getCalibration(image.image_id)
      .then((cal) => {
        if (cancelled || cal.um_per_pixel == null || !Number.isFinite(cal.um_per_pixel) || cal.um_per_pixel <= 0) return
        setUmPerPixelState(cal.um_per_pixel)
        calStore.setCalibration(
          cal.um_per_pixel,
          cal.calibration_id ?? '',
          cal.method,
          cal.confidence,
        )
        if (cal.image_id === image.image_id) calStore.setImageCalibration(image.image_id, cal.um_per_pixel)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [image?.image_id])

  const siblingImages = useMemo(() => {
    if (!image || projectImages.length === 0) return []
    const currentEntry = projectImages.find((pi) => pi.image_id === image.image_id)
    const currentFolder = currentEntry?.folder ?? null
    return projectImages.filter((pi) => (pi.folder ?? null) === currentFolder)
  }, [image, projectImages])

  const currentSiblingIndex = useMemo(() => {
    if (!image) return -1
    return siblingImages.findIndex((pi) => pi.image_id === image.image_id)
  }, [image, siblingImages])

  const hasPrev = currentSiblingIndex > 0
  const hasNext = currentSiblingIndex >= 0 && currentSiblingIndex < siblingImages.length - 1

  // ── Derived State ───────────────────────────────────────────────────
  const [contourOverlay, setContourOverlay] = useState<{ url: string; jobId: string } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let url: string | null = null
    setContourOverlay(null)
    if (measurementDone && measureJob) {
      const jobId = measureJob.id
      getContourOverview(jobId, controller.signal).then(blob => {
        if (!controller.signal.aborted) {
          url = URL.createObjectURL(blob)
          setContourOverlay({ url, jobId })
        }
      }).catch(error => { if (!controller.signal.aborted) setContourError(error.message) })
    }
    return () => { controller.abort(); if (url) URL.revokeObjectURL(url) }
  }, [measurementDone, measureJob?.id])
  // Shown only while it belongs to the current photo's measurement (measurementDone checks the
  // photo), never for the render in which the photo or job has just changed.
  const contourOverlayUrl = contourOverlay && measurementDone && contourOverlay.jobId === measureJob?.id ? contourOverlay.url : null
  const hasContourOverlay = contourOverlayUrl !== null

  // Determine which image to show based on overlay mode
  const viewerSrc = useMemo(() => {
    if (!image) return ''
    if ((overlayMode === 'contours' || overlayMode === 'both') && contourOverlayUrl) {
      return contourOverlayUrl
    }
    return imageUrl(image.image_id, image.filename)
  }, [image, overlayMode, contourOverlayUrl])

  // Show bbox overlay when mode is 'boxes' or 'both' and there are boxes to show
  const showBboxOverlay = !calibrationMode && (overlayMode === 'boxes' || overlayMode === 'both') && (refinement.boxes.length > 0 || (refineMode && editTool === 'draw'))

  // Index of selected box within non-added boxes → maps to measurement CSV row
  const selectedMeasurementIndex = useMemo(() => {
    if (isExperimentalMeasurement) {
      const index = csvData?.findIndex(row => Number(row.detection_id) === selectedContourId) ?? -1
      return index >= 0 ? index : null
    }
    if (!refinement.selectedId) return null
    const box = refinement.boxes.find((b) => b.id === refinement.selectedId)
    if (!box || box.status === 'added') return null
    const origBoxes = refinement.boxes.filter((b) => b.status !== 'added')
    const idx = origBoxes.findIndex((b) => b.id === refinement.selectedId)
    return idx >= 0 ? idx : null
  }, [refinement.selectedId, refinement.boxes, isExperimentalMeasurement, csvData, selectedContourId])

  // ── Effects ─────────────────────────────────────────────────────────

  // Auto-enter refine mode when saved annotations are restored
  useEffect(() => {
    if (refinement.annotationsSaved && !refineMode) {
      setRefineMode(true)
    }
  }, [refinement.annotationsSaved, refineMode])

  // Auto-show boxes overlay when boxes become available (detection result or saved annotations)
  const boxesAutoShownRef = useRef(false)
  useEffect(() => {
    if (refinement.boxes.length > 0 && !boxesAutoShownRef.current && !isExperimentalMeasurement) {
      boxesAutoShownRef.current = true
      setOverlayMode('boxes')
    }
    if (refinement.boxes.length === 0) {
      boxesAutoShownRef.current = false
    }
  }, [refinement.boxes.length, isExperimentalMeasurement])

  // Auto-switch overlay mode when a measurement contour overlay completes
  useEffect(() => {
    if (contourOverlayUrl) setOverlayMode(previous => previous === 'raw' ? 'contours' : previous)
  }, [contourOverlayUrl])

  // Load CSV when measurement completes
  useEffect(() => {
    if (isExperimentalMeasurement && measureJob?.status === 'completed') {
      setCsvData(calibrationMatchesResult ? measureJob.result.measurements as Record<string, any>[] : null)
      return
    }
    setCsvData(null)
  }, [measureJob?.status, measureJob?.result?.csv_path, measureJob?.id, isExperimentalMeasurement, calibrationMatchesResult])

  // Auto-sync job IDs to project
  useEffect(() => {
    if (!currentProjectId || !image || !detectionJobId) return
    if (detectionJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, {
      detection_job_id: detectionJobId,
    }).catch((e) => console.error('[WorkspacePage] Failed to sync detection job to project', e))
  }, [detectionJob?.status, detectionJobId, currentProjectId, image])

  useEffect(() => {
    if (!currentProjectId || !image || !measureJobId) return
    if (measureJob?.status !== 'completed') return
    updateProjectImageJobs(currentProjectId, image.image_id, {
      measurement_job_id: measureJobId,
    }).catch((e) => console.error('[WorkspacePage] Failed to sync measurement job to project', e))
  }, [measureJob?.status, measureJobId, currentProjectId, image])

  const navigateToImage = useCallback(async (projectImg: ProjectImage) => {
    try {
      const imgInfo = await getImage(projectImg.image_id)
      // Reset all state then load new image (batched by React 18)
      workspaceStore.reset()
      workspaceStore.setImage(imgInfo)
      if (projectImg.detection_job_id) workspaceStore.setDetectionJobId(projectImg.detection_job_id)
      if (projectImg.measurement_job_id) workspaceStore.setMeasureJobId(projectImg.measurement_job_id)
      setImageState(imgInfo)
      setDetectionJobIdState(projectImg.detection_job_id)
      setMeasureJobIdState(projectImg.measurement_job_id)
      setCsvData(null)
      setDetectionError(null)
      setMeasureError(null)
      setRefineMode(false)
      setEditTool('select')
      setRefineSaveError(null)
      setOverlayMode('raw')
      setContourJobIdState(null)
      setOrganismContours([])
      setSelectedContourId(null)
      setContourError(null)
      boxesAutoShownRef.current = false
    } catch (e) {
      console.error('[WorkspacePage] Failed to navigate to image', e)
    }
  }, [workspaceStore])

  const handlePrev = useCallback(() => {
    if (!hasPrev) return
    navigateToImage(siblingImages[currentSiblingIndex - 1])
  }, [hasPrev, siblingImages, currentSiblingIndex, navigateToImage])

  const handleNext = useCallback(() => {
    if (!hasNext) return
    navigateToImage(siblingImages[currentSiblingIndex + 1])
  }, [hasNext, siblingImages, currentSiblingIndex, navigateToImage])

  // Keyboard shortcuts: Esc, S, D, H, Delete/Backspace
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      if (e.key === 'Escape') {
        if (editTool !== 'select') {
          setEditTool('select')
          return
        }
        if (refinement.selectedId) {
          refinement.selectBox(null)
          return
        }
        return
      }

      if (!refineMode) return

      if (e.key === 's' || e.key === 'S') {
        setEditTool('select')
        refinement.selectBox(null)
      }
      if (e.key === 'r' || e.key === 'R') {
        setEditTool('reject')
        refinement.selectBox(null)
      }
      if (e.key === 'd' || e.key === 'D') {
        setEditTool('draw')
      }
      if (e.key === 'h' || e.key === 'H') {
        setOverlayMode((prev) => {
          if (prev !== 'raw') { prevOverlayModeRef.current = prev; return 'raw' }
          return prevOverlayModeRef.current
        })
      }

      if (editTool === 'select' && refinement.selectedId) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
          e.preventDefault()
          const sel = refinement.boxes.find((b) => b.id === refinement.selectedId)
          if (sel?.status === 'added') refinement.removeBox(refinement.selectedId)
          else refinement.toggleBox(refinement.selectedId)
        }
      }

      if (!refineMode || editTool !== 'draw') {
        if (e.key === 'ArrowLeft') { e.preventDefault(); handlePrev() }
        if (e.key === 'ArrowRight') { e.preventDefault(); handleNext() }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [editTool, refineMode, refinement, handlePrev, handleNext])

  // ── Handlers ────────────────────────────────────────────────────────

  const handleStartCalibration = () => {
    setCalibrationMode(true)
    setCalibrationPoints([])
    setCalibrationError(null)
    setCalibrationMessage('Please select two points corresponding to 0 mm and 10 mm on the image scale.')
    setRefineMode(false)
    setEditTool('select')
  }

  const handleCancelCalibration = () => {
    setCalibrationMode(false)
    setCalibrationPoints([])
    setCalibrationError(null)
    setCalibrationMessage(null)
  }

  const handleCalibrationClick = async (x: number, y: number) => {
    if (!image || !calibrationMode) return

    const point: Point = [x, y]
    const nextPoints = [...calibrationPoints, point]
    setCalibrationPoints(nextPoints)
    setCalibrationError(null)

    if (nextPoints.length === 1) {
      setCalibrationMessage('First point selected. Click the 10 mm mark.')
      return
    }

    if (nextPoints.length !== 2) return


    try {
      const result = await manualCalibrate(image.image_id, nextPoints[0], nextPoints[1], 10)
      if (activeImageRef.current !== image.image_id) return
      if (result.um_per_pixel == null) throw new Error(result.error || 'Calibration failed')
      setUmPerPixelState(result.um_per_pixel)
      calStore.setCalibration(
        result.um_per_pixel,
        result.calibration_id ?? '',
        result.method,
        result.confidence,
      )
      calStore.setImageCalibration(image.image_id, result.um_per_pixel)
      setCalibrationMode(false)
      setCalibrationMessage(
        `Saved scale: ${result.um_per_pixel.toFixed(4)} μm/px (${result.ruler_px?.toFixed(1)} px over 10 mm).`,
      )
    } catch (e: any) {
      setCalibrationError(e.message || 'Calibration failed')
      setCalibrationMessage(null)
      setCalibrationPoints([])
    }
  }

  const handleRunDetection = async (config?: {
    tileSize: number
    overlap: number
    confidence: number
    iouThreshold: number
    device: string
  }) => {
    if (!image) return
    setDetectionError(null)

    const detectionConfig = config
      ? {
          conf: config.confidence,
          iou: config.iouThreshold,
          tile_size: config.tileSize,
          overlap: config.overlap,
          device: config.device === 'auto' ? '0' : config.device,
        }
      : { device }

    try {
      const res = await runDetection({
        image_id: image.image_id,
        ...detectionConfig,
      })
      setDetectionJobId(res.job_id)
    } catch (e: any) {
      setDetectionError(e.message)
    }
  }

  const handleDoneEditing = async () => {
    const effectiveJobId = refinement.restoredSourceJobId || detectionJobId || ''
    if (!image) return
    setRefineSaveError(null)
    try {
      await refinement.saveAnnotations(image.image_id, image.filename, effectiveJobId)
      setRefineMode(false)
      setEditTool('select')
    } catch (e: any) {
      setRefineSaveError(e.message)
    }
  }

  const UNSAVED_PREFIX = 'Spremite izmjene pravokutnika'
  useEffect(() => {
    if (!refinement.hasUnsavedEdits) setContourError(prev => prev?.startsWith(UNSAVED_PREFIX) ? null : prev)
  }, [refinement.hasUnsavedEdits])

  const [mergeMessage, setMergeMessage] = useState<string | null>(null)
  useEffect(() => { setMergeMessage(null) }, [image?.image_id, detectionJobId])
  const handleMergeDetections = async () => {
    const summary = await refinement.mergeDetections()
    if (!summary) return
    const kept = summary.added_kept + summary.rejected_kept + summary.previous_accepted_kept
    setMergeMessage(`${summary.new_kept} boxes from the new run; ${kept} kept from your review. Check and Save (measuring waits for Save).`)
    setRefineMode(true)
    setOverlayMode('boxes')
  }

  const handleRunMeasurement = async () => {
    setMeasureError(null)
    await submitContours()
  }

  const handleReset = () => {
    workspaceStore.reset()
    setImageState(null)
    setDetectionJobIdState(null)
    setMeasureJobIdState(null)
    setCsvData(null)
    setDetectionError(null)
    setMeasureError(null)
    setRefineMode(false)
    setEditTool('select')
    setRefineSaveError(null)
    setOverlayMode('raw')
    setContourJobIdState(null)
    setOrganismContours([])
    setSelectedContourId(null)
    setContourError(null)
  }

  const handleRowClick = useCallback(
    (rowIndex: number) => {
      if (isExperimentalMeasurement) {
        const id = Number(csvData?.[rowIndex]?.detection_id)
        if (Number.isFinite(id)) { setSelectedContourId(id); setOverlayMode('contours') }
        return
      }
      const origBoxes = refinement.boxes.filter((b) => b.status !== 'added')
      const box = origBoxes[rowIndex]
      if (!box) return
      refinement.selectBox(box.id)
    },
    [refinement, isExperimentalMeasurement, csvData]
  )

  const handleExport = (format: 'image' | 'csv' | 'excel') => {
    if (!measureJob || !measureJob.result) return
    if (format !== 'image' && !isExperimentalMeasurement) {
      setContourError('Najprije izračunajte rezultate središnjom linijom.')
      return
    }
    if (format !== 'image' && isExperimentalMeasurement && !calibrationMatchesResult) {
      setContourError('Mjerenja se ponovno računaju za novu kalibraciju. Pričekajte dovršetak prije izvoza.')
      return
    }

    if (format === 'image') {
      // Download the current view
      const link = document.createElement('a')
      link.href = viewerSrc
      link.download = image?.filename ?? 'image.jpg'
      link.click()
    } else if (format === 'csv' && measureJob.result.csv_path) {
      const filename = basename(measureJob.result.csv_path)!
      const url = outputFileUrl(measureJob.id, filename)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
    } else if (format === 'excel' && measureJob.result.excel_path) {
      const filename = basename(measureJob.result.excel_path)!
      const url = outputFileUrl(measureJob.id, filename)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
    }
  }


  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      {/* Top bar */}
      <header className="shrink-0 bg-white border-b px-5 h-12 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <span className="font-semibold text-gray-900">Collembola</span>
          <nav className="flex items-center gap-1">
            {currentProjectId && (
              <Link
                to={`/projects/${currentProjectId}`}
                className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
              >
                ← {currentProjectName ?? 'Project'}
              </Link>
            )}
            <Link
              to="/projects"
              className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
            >
              Projects
            </Link>
            <Link
              to="/workspace"
              className="text-sm px-3 py-1 rounded-md bg-gray-100 text-gray-900 font-medium"
            >
              Workspace
            </Link>
            {role === 'admin' && (
              <Link
                to="/finetune"
                className="text-sm px-3 py-1 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 transition-colors"
              >
                Fine-Tune
              </Link>
            )}
            <Link to="/detector-dataset" className="text-sm px-3 py-1 text-teal-700">Dataset za trening</Link>
          </nav>
          {image && (
            <span
              className="text-xs text-gray-400 truncate max-w-[200px] hidden sm:block"
              title={image.filename}
            >
              {image.filename}
            </span>
          )}
        </div>
        <button
          onClick={() => {
            logout()
            navigate('/login', { replace: true })
          }}
          className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700"
        >
          <LogOut size={14} />
          Sign out
        </button>
      </header>

      {!image ? (
        /* ── Empty state ── */
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-lg space-y-4">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-gray-900">
                Load an image to get started
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                Upload a microscope image or reference one already on the server
              </p>
            </div>
            <ImageUploader onUploaded={setImage} />
            <ExperimentalContourAction busy={contourBusy} job={contourJob ?? null} error={contourError}
              calibration={contourCalibration} onRun={() => { void submitContours() }} />
          </div>
        </div>
      ) : (
        /* ── Workspace ── */
        <div className="flex flex-1 overflow-hidden">
          {/* ── Sidebar ── */}
          {sidebarCollapsed ? (
            /* Collapsed strip */
            <aside className="w-10 shrink-0 bg-white border-r flex flex-col items-center pt-3 gap-4">
              <button
                onClick={() => setSidebarCollapsed(false)}
                className="text-gray-400 hover:text-gray-700 p-1 rounded hover:bg-gray-100"
                title="Expand sidebar"
              >
                <ChevronRight size={16} />
              </button>
              {/* Mini status dots */}
              <div className="flex flex-col gap-1.5 mt-1">
                {[
                  { done: calStore.umPerPixel != null, label: 'Scale' },
                  { done: detectionDone, label: 'Detect' },
                  { done: refinement.annotationsSaved, label: 'Edit' },
                  { done: measurementDone, label: 'Measure' },
                ].map((s) => (
                  <div
                    key={s.label}
                    className={`w-2.5 h-2.5 rounded-full ${s.done ? 'bg-green-500' : 'bg-gray-200'}`}
                    title={s.label}
                  />
                ))}
              </div>
            </aside>
          ) : (
            /* Expanded sidebar */
            <aside className="w-64 md:w-64 sm:w-full shrink-0 bg-white border-r flex flex-col overflow-hidden max-w-full">
              {/* Sidebar header */}
              <div className="px-4 py-3 border-b flex items-center justify-between">
                <h2 className="font-semibold text-gray-900 text-sm">Workspace</h2>
                <button
                  onClick={() => setSidebarCollapsed(true)}
                  className="text-gray-400 hover:text-gray-700 p-1 rounded hover:bg-gray-100"
                  title="Collapse sidebar"
                >
                  <ChevronRight size={16} className="rotate-180" />
                </button>
              </div>

              <WorkspaceSidebar
                imageFilename={image.filename}
                imageWidth={image.width}
                imageHeight={image.height}
                umPerPixel={umPerPixel}
                setUmPerPixel={setUmPerPixel}
                calibrated={contourCalibration != null}
                calibrationMode={calibrationMode}
                calibrationPointCount={calibrationPoints.length}
                calibrationError={calibrationError}
                calibrationMessage={calibrationMessage}
                onStartCalibration={handleStartCalibration}
                onCancelCalibration={handleCancelCalibration}
                detectionDone={detectionDone}
                detectionJob={detectionJob ?? null}
                detectionError={detectionError}
                onRunDetection={() => handleRunDetection()}
                onAdvancedSettings={() => setModalOpen('advanced-detection')}
                annotationsSaved={refinement.annotationsSaved}
                boxCount={refinement.boxes.length}
                detectorDatasetAction={<SaveDetectorDataset key={image.image_id} imageId={image.image_id}
                  sourceJobId={refinement.restoredSourceJobId || detectionJobId}
                  boxes={refinement.boxes} disabled={refinement.isLoading || !!refinement.error || (!!detectionJobId && !detectionDone)} />}
                measurementDone={measurementDone}
                measureJob={measureJob ?? null}
                measureError={needsCurrentMeasurement ? 'Za prikaz rezultata pokrenite mjerenje središnjom linijom. Odgovarajuće spremljene maske ponovno će se koristiti.' : measureError}
                onRunMeasurement={handleRunMeasurement}
                csvDataLength={csvData?.length ?? 0}
                onViewResults={() => setShowMeasurementModal(true)}
                onReset={handleReset}
                contourBusy={contourBusy}
                contourJob={contourJob ?? null}
                contourError={contourError}
                contourCalibration={contourCalibration}
                onRunContours={() => { void submitContours() }}
                volumeSummary={volumeSummary}
                contourCount={isExperimentalMeasurement ? Number(measureJob?.result.num_organisms ?? 0) : 0}
              />
            </aside>
          )}

          {/* ── Main Content ── */}
          <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
            {/* Toolbar row */}
            <div className="flex items-center border-b">
              <div className="flex-1">
                <ViewerToolbar
                  overlayMode={overlayMode}
                  onOverlayChange={setOverlayMode}
                  availableOverlays={{
                    boxes: detectionDone,
                    contours: hasContourOverlay,
                  }}
                  onExport={handleExport}
                  measurementDone={measurementDone}
                />
              </div>
              {siblingImages.length > 1 && (
                <div className="flex items-center gap-0.5 px-2 border-l shrink-0">
                  <button
                    onClick={handlePrev}
                    disabled={!hasPrev}
                    title="Previous image (←)"
                    className="p-1 rounded text-gray-500 hover:text-gray-800 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-default transition-colors"
                  >
                    <ChevronLeft size={15} />
                  </button>
                  <span className="text-xs text-gray-400 tabular-nums px-0.5">
                    {currentSiblingIndex + 1}/{siblingImages.length}
                  </span>
                  <button
                    onClick={handleNext}
                    disabled={!hasNext}
                    title="Next image (→)"
                    className="p-1 rounded text-gray-500 hover:text-gray-800 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-default transition-colors"
                  >
                    <ChevronRight size={15} />
                  </button>
                </div>
              )}
              {measurementDone && (
                <button
                  onClick={() => setShowMeasurementModal(true)}
                  className="flex items-center gap-1.5 text-sm px-3 py-1.5 mr-3 border border-gray-300 rounded-lg bg-white hover:bg-gray-50 text-gray-700 font-medium transition-colors shrink-0"
                >
                  <Table2 size={14} />
                  View Results
                </button>
              )}
            </div>

            {/* Image viewer fills remaining space */}
            <div className="flex-1 min-h-0 relative overflow-hidden">
              <ImageViewer
                src={viewerSrc}
                alt={image.filename}
                className="h-full"
                focusBox={selectedContourBox}
                onImageClick={calibrationMode ? handleCalibrationClick : undefined}
                transformOverlay={(showBboxOverlay || calibrationPoints.length > 0) ? (
                  <>
                    {showBboxOverlay && (
                      <BboxOverlay
                        boxes={refinement.boxes}
                        imageWidth={image.width}
                        imageHeight={image.height}
                        selectedId={refinement.selectedId}
                        onBoxClick={refineMode && editTool === 'reject'
                          ? (id) => {
                              const box = refinement.boxes.find((b) => b.id === id)
                              if (box?.status === 'added') refinement.removeBox(id)
                              else refinement.toggleBox(id)
                            }
                          : refinement.selectBox}
                        drawingBox={refinement.drawingBox}
                        mode={refineMode && editTool === 'draw' ? 'draw' : 'review'}
                        onDrawStart={refineMode ? refinement.startDraw : undefined}
                        onDrawMove={refineMode ? refinement.updateDraw : undefined}
                        onDrawEnd={refineMode ? refinement.commitDraw : undefined}
                      />
                    )}
                    {calibrationPoints.length > 0 && (
                      <svg
                        viewBox={`0 0 ${image.width} ${image.height}`}
                        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                      >
                        {calibrationPoints.length === 2 && (
                          <line
                            x1={calibrationPoints[0][0]}
                            y1={calibrationPoints[0][1]}
                            x2={calibrationPoints[1][0]}
                            y2={calibrationPoints[1][1]}
                            stroke="#f59e0b"
                            strokeWidth={Math.max(4, image.width / 1500)}
                            strokeDasharray={`${Math.max(8, image.width / 700)} ${Math.max(6, image.width / 900)}`}
                          />
                        )}
                        {calibrationPoints.map(([x, y], i) => (
                          <g key={`${x}-${y}-${i}`}>
                            <circle cx={x} cy={y} r={Math.max(10, image.width / 700)} fill="#f59e0b" opacity="0.9" />
                            <circle cx={x} cy={y} r={Math.max(18, image.width / 420)} fill="none" stroke="#f59e0b" strokeWidth={Math.max(3, image.width / 2200)} />
                          </g>
                        ))}
                      </svg>
                    )}
                  </>
                ) : undefined}
                disablePan={calibrationMode || (refineMode && editTool === 'draw')}
              />

              {calibrationMode && (
                <div className="absolute left-1/2 top-3 z-20 -translate-x-1/2 rounded-full border border-amber-300 bg-amber-50 px-4 py-2 text-sm font-medium text-amber-800 shadow-sm">
                  Click {calibrationPoints.length === 0 ? '0 mm' : '10 mm'} mark on the scale bar
                </div>
              )}

              {/* Floating annotation toolbar — always visible once detection is done */}
              {image && !calibrationMode && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 flex flex-col gap-0.5 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg p-1.5">
                  {refineMode ? (
                    <>
                      {/* Edit tools */}
                      {([
                        { tool: 'select', label: 'Select', key: 'S', Icon: MousePointer2 },
                        { tool: 'reject', label: 'Invalid', key: 'R', Icon: XCircle },
                        { tool: 'draw',   label: 'Draw',   key: 'D', Icon: Square },
                      ] as const).map(({ tool, label, key, Icon }) => (
                        <button
                          key={tool}
                          title={`${label} (${key})`}
                          onClick={() => {
                            setEditTool(tool)
                            refinement.selectBox(null)
                          }}
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between gap-3 ${
                            editTool === tool
                              ? tool === 'reject'
                                ? 'bg-red-500 text-white'
                                : 'bg-amber-500 text-white'
                              : 'text-gray-600 hover:bg-gray-100'
                          }`}
                        >
                          <span className="flex items-center gap-1.5">
                            <Icon size={12} />
                            {label}
                          </span>
                          <kbd className={`text-[9px] px-1 py-0.5 rounded border font-mono ${
                            editTool === tool
                              ? tool === 'reject'
                                ? 'border-red-300 bg-red-400/40'
                                : 'border-amber-300 bg-amber-400/40'
                              : 'border-gray-200 bg-gray-50 text-gray-400'
                          }`}>
                            {key}
                          </kbd>
                        </button>
                      ))}

                      <div className="border-t border-gray-200 my-0.5" />

                      {/* Hide/Show toggle */}
                      <button
                        onClick={() => setOverlayMode((prev) => {
                          if (prev !== 'raw') { prevOverlayModeRef.current = prev; return 'raw' }
                          return prevOverlayModeRef.current
                        })}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-between gap-3 ${
                          overlayMode !== 'raw'
                            ? 'text-gray-600 hover:bg-gray-100'
                            : 'bg-amber-50 text-amber-700 hover:bg-amber-100'
                        }`}
                      >
                        <span>{overlayMode !== 'raw' ? 'Hide' : 'Show'}</span>
                        <kbd className={`text-[9px] px-1 py-0.5 rounded border font-mono ${
                          overlayMode === 'raw' ? 'border-amber-300 bg-amber-400/40' : 'border-gray-200 bg-gray-50 text-gray-400'
                        }`}>
                          H
                        </kbd>
                      </button>

                      <div className="border-t border-gray-200 my-0.5" />

                      {/* Counts */}
                      <div className="flex items-center justify-between gap-2 px-2 py-0.5 text-[10px]">
                        <span className="text-green-600">{refinement.acceptedCount} ✓</span>
                        <span className="text-red-500">{refinement.rejectedCount} ✗</span>
                        <span className="text-blue-600">{refinement.addedCount} +</span>
                      </div>

                      <div className="border-t border-gray-200 my-0.5" />

                      {editTool === 'select' && refinement.boxes.filter(b => b.id === refinement.selectedId && b.status !== 'rejected').map(box =>
                        <BoxCoordinateEditor key={JSON.stringify([box.id, box.x1, box.y1, box.x2, box.y2])} box={box} width={image.width} height={image.height} onApply={refinement.updateBox} />)}

                      {/* Save & Exit */}
                      <button
                        onClick={handleDoneEditing}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-green-600 text-white hover:bg-green-700 transition-colors flex items-center gap-1.5"
                      >
                        <CheckCircle2 size={12} />
                        Save
                      </button>

                      {/* Fine-tune shortcut (admin only) */}
                      {role === 'admin' && (
                        <button
                          onClick={() => navigate('/finetune')}
                          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-violet-600 text-white hover:bg-violet-700 transition-colors"
                        >
                          Fine-Tune
                        </button>
                      )}

                      {/* Save error */}
                      {refineSaveError && (
                        <p className="text-[10px] text-red-600 px-2 max-w-[140px] leading-tight">{refineSaveError}</p>
                      )}
                    </>
                  ) : (
                    <>
                      {/* Entry state */}
                      {refinement.hasUnsavedEdits && (
                        <div className="text-[10px] text-amber-700 px-2 py-1" title="Measuring uses saved boxes only">
                          Unsaved box edits: Save before measuring
                        </div>
                      )}
                      {refinement.annotationsSaved && (
                        <div className="flex items-center gap-1 text-[10px] text-green-700 px-2 py-1">
                          <CheckCircle2 size={10} />
                          Annotations saved
                        </div>
                      )}
                      {refinement.canMergeDetections && (
                        <button
                          onClick={handleMergeDetections}
                          disabled={refinement.isMerging}
                          title="Replace the model boxes of the saved annotations with the current detection run. Your added and rejected boxes are kept; review and Save to keep the result."
                          className="px-3 py-1.5 rounded-lg text-xs font-medium text-blue-700 hover:bg-blue-50 transition-colors disabled:opacity-50"
                        >
                          {refinement.isMerging ? 'Merging…' : 'Merge new detections'}
                        </button>
                      )}
                      {mergeMessage && (
                        <p className="text-[10px] text-blue-700 px-2 max-w-[160px] leading-tight">{mergeMessage}</p>
                      )}
                      <button
                        onClick={() => { setRefineMode(true); setOverlayMode('boxes') }}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium text-gray-700 hover:bg-gray-100 transition-colors flex items-center gap-1.5"
                      >
                        <Pencil size={12} />
                        {refinement.annotationsSaved ? 'Edit Annotations' : 'Edit Detections'}
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Modals ── */}
      <AdvancedDetectionModal
        isOpen={modalOpen === 'advanced-detection'}
        onClose={() => setModalOpen(null)}
        onSubmit={(config) => {
          handleRunDetection(config)
          setModalOpen(null)
        }}
        initialConfig={{ tileSize, overlap, confidence: conf, iouThreshold: 0.5, device }}
      />

      <MeasurementModal
        isOpen={showMeasurementModal}
        onClose={() => setShowMeasurementModal(false)}
        data={csvData ?? []}
        selectedIndex={selectedMeasurementIndex}
        onRowClick={handleRowClick}
        onExport={(fmt) => handleExport(fmt)}
        measurementDone={measurementDone && calibrationMatchesResult}
        selectionPreview={isExperimentalMeasurement && image && selectedContour ? (
          <ImageViewer src={imageUrl(image.image_id, image.filename)} alt="Odabrana exp03 kontura" className="h-full" focusBox={selectedContourBox}
            transformOverlay={<ContourHighlight contour={selectedContour} width={image.width} height={image.height} />} />
        ) : undefined}
      />
    </div>
  )
}
