import { create } from 'zustand'

interface CalibrationState {
  umPerPixel: number | null
  calibrationId: string | null
  method: string | null
  confidence: number | null
  setCalibration: (umPerPixel: number, calibrationId: string, method: string, confidence: number) => void
  clear: () => void
}

export const useCalibrationStore = create<CalibrationState>((set) => ({
  umPerPixel: null,
  calibrationId: null,
  method: null,
  confidence: null,
  setCalibration: (umPerPixel, calibrationId, method, confidence) =>
    set({ umPerPixel, calibrationId, method, confidence }),
  clear: () => set({ umPerPixel: null, calibrationId: null, method: null, confidence: null }),
}))
