import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ImageInfo } from '../api/types'

interface WorkspaceState {
  image: ImageInfo | null
  detectionJobId: string | null
  measureJobId: string | null
  setImage: (image: ImageInfo | null) => void
  setDetectionJobId: (id: string | null) => void
  setMeasureJobId: (id: string | null) => void
  reset: () => void
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      image: null,
      detectionJobId: null,
      measureJobId: null,
      setImage: (image) => set({ image }),
      setDetectionJobId: (id) => set({ detectionJobId: id }),
      setMeasureJobId: (id) => set({ measureJobId: id }),
      reset: () => set({ image: null, detectionJobId: null, measureJobId: null }),
    }),
    { name: 'workspace' },
  ),
)
