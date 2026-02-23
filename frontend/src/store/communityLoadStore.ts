import { create } from 'zustand'
import type { AnnotatedBox } from '../api/types'

interface CommunityLoadState {
  boxes: AnnotatedBox[]
  imageName: string
  submittedBy: string
  numDetections: number
  setPending: (boxes: AnnotatedBox[], imageName: string, submittedBy: string) => void
  clear: () => void
}

export const useCommunityLoadStore = create<CommunityLoadState>((set) => ({
  boxes: [],
  imageName: '',
  submittedBy: '',
  numDetections: 0,
  setPending: (boxes, imageName, submittedBy) =>
    set({ boxes, imageName, submittedBy, numDetections: boxes.length }),
  clear: () => set({ boxes: [], imageName: '', submittedBy: '', numDetections: 0 }),
}))
