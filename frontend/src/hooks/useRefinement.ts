import { useState, useEffect, useCallback } from 'react'
import type { AnnotatedBox, AnnotationFile, MergeSummary } from '../api/types'
import {
  getDetectionBoxes,
  getAnnotations,
  mergeDetections as mergeDetectionsApi,
  saveAnnotations as saveAnnotationsApi,
} from '../api/client'

const MIN_BOX_PX = 5 // ignore accidental drags smaller than this threshold
const NO_BOXES: AnnotatedBox[] = []

interface DrawingBox {
  x1: number; y1: number; x2: number; y2: number
}

interface UseRefinementResult {
  boxes: AnnotatedBox[]
  isLoading: boolean
  error: string | null
  toggleBox: (id: string) => void
  removeBox: (id: string) => void
  loadBoxes: (boxes: AnnotatedBox[]) => void
  updateBox: (id: string, coords: DrawingBox) => void
  selectedId: string | null
  selectBox: (id: string | null) => void
  drawingBox: DrawingBox | null
  startDraw: (x: number, y: number) => void
  updateDraw: (x: number, y: number) => void
  commitDraw: (x: number, y: number) => void
  saveAnnotations: (imageId: string, imageFilename: string, detectionJobId: string) => Promise<void>
  isSaving: boolean
  annotationsSaved: boolean
  /**
   * The boxes on screen were changed (edit, merge, community load) and not saved yet. Measuring would
   * ignore these changes (it uses the saved file or the detection job), so it must wait for Save.
   */
  hasUnsavedEdits: boolean
  /** source_job_id from the annotation file that was auto-restored (null if not from file) */
  restoredSourceJobId: string | null
  /**
   * The saved annotations were reviewed for another detection job than the current one: the reviewer's
   * work is shown, and `mergeDetections` can bring in the current job's boxes.
   */
  canMergeDetections: boolean
  /** Replace model boxes with the current detection job's boxes, keeping added/rejected boxes (unsaved). */
  mergeDetections: () => Promise<MergeSummary | null>
  isMerging: boolean
  acceptedCount: number
  rejectedCount: number
  addedCount: number
}

export function useRefinement(
  imageId: string | null,
  detectionJobId: string | null,
  detectionDone: boolean,
): UseRefinementResult {
  const [boxes, setBoxes] = useState<AnnotatedBox[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [drawingBox, setDrawingBox] = useState<DrawingBox | null>(null)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const [annotationsSaved, setAnnotationsSaved] = useState(false)
  const [hasUnsavedEdits, setHasUnsavedEdits] = useState(false)
  const [restoredSourceJobId, setRestoredSourceJobId] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  // Photo the box state belongs to. Everything returned below is gated on it, so the
  // render in which `imageId` changes can never pair the new photo with old boxes.
  const [boxesImageId, setBoxesImageId] = useState<string | null>(null)
  const current = imageId != null && boxesImageId === imageId

  const [isMerging, setIsMerging] = useState(false)

  // Load boxes whenever imageId or detectionJobId changes.
  // Priority: human-saved annotations > current detection job > annotations written without review.
  // A file saved from the editor (saved_at) holds the reviewer's work and is always preferred. A file
  // without saved_at was written from raw detections (e.g. at project import); it only stands in for
  // its own detection job, so a newer job (e.g. a re-run with the current detector) replaces it.
  useEffect(() => {
    if (!imageId) {
      setBoxes([])
      setBoxesImageId(null)
      setSelectedId(null)
      setAnnotationsSaved(false)
      setHasUnsavedEdits(false)
      setRestoredSourceJobId(null)
      return
    }

    let cancelled = false
    setIsLoading(true)
    setError(null)
    setBoxes([])
    setBoxesImageId(imageId)
    setSelectedId(null)
    setAnnotationsSaved(false)
    setHasUnsavedEdits(false)
    setRestoredSourceJobId(null)

    async function load() {
      try {
        // Always try saved annotations first — they represent human-reviewed work
        // and take priority over any detection job's raw output.
        let restored = false
        try {
          const ann = await getAnnotations(imageId!)
          // Only a finished job replaces the file: while a re-run is pending, or if it failed, the file stays.
          const supersededByNewerJob = !ann.saved_at && !!detectionJobId && detectionDone && ann.source_job_id !== detectionJobId
          // A file written for another photo is never shown on this one.
          if (ann.image_id === imageId && ann.boxes.length > 0 && !supersededByNewerJob && !cancelled) {
            setBoxes(ann.boxes)
            setAnnotationsSaved(true)
            setRestoredSourceJobId(ann.source_job_id)
            restored = true
          }
        } catch {
          // No saved annotations — fall through to detection boxes
        }

        if (!restored && detectionJobId && !cancelled) {
          const result = await getDetectionBoxes(detectionJobId!)
          if (!cancelled && result.image_id && result.image_id !== imageId) {
            setError('Detekcije pripadaju drugoj fotografiji; pravokutnici nisu prikazani.')
          } else if (!cancelled) {
            setBoxes(result.boxes)
          }
        }
      } catch (e: any) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [detectionJobId, detectionDone, imageId])

  const toggleBox = useCallback((id: string) => {
    setAnnotationsSaved(false)
    setHasUnsavedEdits(true)
    setBoxes((prev) =>
      prev.map((b) =>
        b.id === id
          ? { ...b, status: b.status === 'rejected' ? 'accepted' : 'rejected' }
          : b,
      ),
    )
  }, [])

  const removeBox = useCallback((id: string) => {
    setBoxes((prev) => prev.filter((b) => b.id !== id))
    setSelectedId((prev) => (prev === id ? null : prev))
    setAnnotationsSaved(false)
    setHasUnsavedEdits(true)
  }, [])

  const selectBox = useCallback((id: string | null) => {
    setSelectedId(id)
  }, [])

  const loadBoxes = useCallback((incoming: AnnotatedBox[]) => {
    setBoxes(incoming)
    setAnnotationsSaved(false)
    setHasUnsavedEdits(true)
  }, [])

  const updateBox = useCallback((id: string, coords: DrawingBox) => {
    setBoxes(prev => prev.map(box => box.id === id ? { ...box, ...coords } : box))
    setAnnotationsSaved(false)
    setHasUnsavedEdits(true)
  }, [])

  const startDraw = useCallback((x: number, y: number) => {
    setDrawStart({ x, y })
    setDrawingBox({ x1: x, y1: y, x2: x, y2: y })
  }, [])

  const updateDraw = useCallback((x: number, y: number) => {
    setDrawingBox((prev) => prev ? { ...prev, x2: x, y2: y } : null)
  }, [])

  const commitDraw = useCallback((x: number, y: number) => {
    setDrawingBox(null)
    setDrawStart(null)
    if (!drawStart) return
    const x1 = Math.min(drawStart.x, x)
    const y1 = Math.min(drawStart.y, y)
    const x2 = Math.max(drawStart.x, x)
    const y2 = Math.max(drawStart.y, y)
    // Ignore tiny accidental drags
    if (x2 - x1 < MIN_BOX_PX || y2 - y1 < MIN_BOX_PX) return
    const newBox: AnnotatedBox = {
      id: `added-${crypto.randomUUID()}`,
      x1, y1, x2, y2,
      conf: 1.0,
      status: 'added',
    }
    setBoxes((prev) => [...prev, newBox])
    setAnnotationsSaved(false)
    setHasUnsavedEdits(true)
  }, [drawStart])

  const saveAnnotations = useCallback(async (
    imageId: string,
    imageFilename: string,
    detectionJobId: string,
  ) => {
    if (imageId !== boxesImageId) throw new Error('Oznake pripadaju drugoj fotografiji; nisu spremljene.')
    setIsSaving(true)
    try {
      const payload: AnnotationFile = {
        image_id: imageId,
        image_filename: imageFilename,
        source_job_id: detectionJobId,
        created_at: new Date().toISOString(),
        boxes,
      }
      await saveAnnotationsApi(imageId, payload)
      setAnnotationsSaved(true)
      setHasUnsavedEdits(false)
    } finally {
      setIsSaving(false)
    }
  }, [boxes, boxesImageId])

  const canMergeDetections = current && restoredSourceJobId != null && detectionJobId != null
    && detectionDone && restoredSourceJobId !== detectionJobId

  const mergeDetections = useCallback(async (): Promise<MergeSummary | null> => {
    if (!imageId || imageId !== boxesImageId || !detectionJobId) return null
    setIsMerging(true)
    setError(null)
    try {
      const merged = await mergeDetectionsApi(imageId, detectionJobId)
      if (merged.image_id !== imageId) throw new Error('Detekcije pripadaju drugoj fotografiji.')
      setBoxes(merged.boxes)
      setSelectedId(null)
      setAnnotationsSaved(false)
      setHasUnsavedEdits(true)
      // Saving now records the merged-in job as the source of the model boxes.
      setRestoredSourceJobId(merged.source_job_id)
      return merged.summary
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      return null
    } finally {
      setIsMerging(false)
    }
  }, [imageId, boxesImageId, detectionJobId])

  const visibleBoxes = current ? boxes : NO_BOXES
  return {
    boxes: visibleBoxes,
    isLoading,
    error,
    toggleBox,
    removeBox,
    loadBoxes,
    updateBox,
    selectedId: current ? selectedId : null,
    selectBox,
    drawingBox,
    startDraw,
    updateDraw,
    commitDraw,
    saveAnnotations,
    isSaving,
    annotationsSaved: current && annotationsSaved,
    hasUnsavedEdits: current && hasUnsavedEdits,
    restoredSourceJobId: current ? restoredSourceJobId : null,
    canMergeDetections,
    mergeDetections,
    isMerging,
    acceptedCount: visibleBoxes.filter((b) => b.status === 'accepted').length,
    rejectedCount: visibleBoxes.filter((b) => b.status === 'rejected').length,
    addedCount: visibleBoxes.filter((b) => b.status === 'added').length,
  }
}
