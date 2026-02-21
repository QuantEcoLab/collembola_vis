import { create } from 'zustand'
import type { Job } from '../api/types'

interface JobState {
  jobs: Record<string, Job>
  updateJob: (job: Job) => void
  getJob: (id: string) => Job | undefined
}

export const useJobStore = create<JobState>((set, get) => ({
  jobs: {},
  updateJob: (job) => set((state) => ({ jobs: { ...state.jobs, [job.id]: job } })),
  getJob: (id) => get().jobs[id],
}))
