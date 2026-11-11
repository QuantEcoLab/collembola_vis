import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ImageInfo } from '../api/types'

interface WorkspaceState {
  image: ImageInfo | null
  detectionJobId: string | null
  measureJobId: string | null
  contourJobId: string | null
  setImage: (image: ImageInfo | null) => void
  setDetectionJobId: (id: string | null) => void
  setMeasureJobId: (id: string | null) => void
  setContourJobId: (id: string | null) => void
  reset: () => void
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      image: null,
      detectionJobId: null,
      measureJobId: null,
      contourJobId: null,
      setImage: (image) => set({ image }),
      setDetectionJobId: (id) => set({ detectionJobId: id }),
      setMeasureJobId: (id) => set({ measureJobId: id }),
      setContourJobId: (id) => set({ contourJobId: id }),
      reset: () => set({ image: null, detectionJobId: null, measureJobId: null, contourJobId: null }),
    }),
    { name: 'workspace' },
  ),
)
