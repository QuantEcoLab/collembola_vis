import { useState, useEffect, useCallback } from 'react'
import type { AnnotatedBox, AnnotationFile } from '../api/types'
import {
  getDetectionBoxes,
  getAnnotations,
  saveAnnotations as saveAnnotationsApi,
} from '../api/client'

const MIN_BOX_PX = 5 // ignore accidental drags smaller than this threshold

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
  selectedId: string | null
  selectBox: (id: string | null) => void
  drawingBox: DrawingBox | null
  startDraw: (x: number, y: number) => void
  updateDraw: (x: number, y: number) => void
  commitDraw: (x: number, y: number) => void
  saveAnnotations: (imageId: string, imageFilename: string, detectionJobId: string) => Promise<void>
  isSaving: boolean
  annotationsSaved: boolean
  /** source_job_id from the annotation file that was auto-restored (null if not from file) */
  restoredSourceJobId: string | null
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
  const [restoredSourceJobId, setRestoredSourceJobId] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Load boxes whenever imageId or detectionJobId changes.
  // Priority: saved annotations > detection boxes.
  // Saved annotations represent human-reviewed work and are always preferred,
  // regardless of which detection job they originated from.
  useEffect(() => {
    if (!imageId) {
      setBoxes([])
      setSelectedId(null)
      setAnnotationsSaved(false)
      setRestoredSourceJobId(null)
      return
    }

    let cancelled = false
    setIsLoading(true)
    setError(null)
    setBoxes([])
    setSelectedId(null)
    setAnnotationsSaved(false)
    setRestoredSourceJobId(null)

    async function load() {
      try {
        // Always try saved annotations first — they represent human-reviewed work
        // and take priority over any detection job's raw output.
        let restored = false
        try {
          const ann = await getAnnotations(imageId!)
          if (ann.boxes.length > 0 && !cancelled) {
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
          if (!cancelled) setBoxes(result.boxes)
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
  }, [])

  const selectBox = useCallback((id: string | null) => {
    setSelectedId(id)
  }, [])

  const loadBoxes = useCallback((incoming: AnnotatedBox[]) => {
    setBoxes(incoming)
    setAnnotationsSaved(false)
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
      id: `added-${Date.now()}`,
      x1, y1, x2, y2,
      conf: 1.0,
      status: 'added',
    }
    setBoxes((prev) => [...prev, newBox])
    setAnnotationsSaved(false)
  }, [drawStart])

  const saveAnnotations = useCallback(async (
    imageId: string,
    imageFilename: string,
    detectionJobId: string,
  ) => {
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
    } finally {
      setIsSaving(false)
    }
  }, [boxes])

  return {
    boxes,
    isLoading,
    error,
    toggleBox,
    removeBox,
    loadBoxes,
    selectedId,
    selectBox,
    drawingBox,
    startDraw,
    updateDraw,
    commitDraw,
    saveAnnotations,
    isSaving,
    annotationsSaved,
    restoredSourceJobId,
    acceptedCount: boxes.filter((b) => b.status === 'accepted').length,
    rejectedCount: boxes.filter((b) => b.status === 'rejected').length,
    addedCount: boxes.filter((b) => b.status === 'added').length,
  }
}
