import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface CalibrationState {
  umPerPixel: number | null
  rulerMm: number
  calibrationId: string | null
  method: string | null
  confidence: number | null
  imageUmPerPixel: Record<string, number>
  setImageCalibration: (imageId: string, value: number | null) => void
  setCalibration: (umPerPixel: number, calibrationId: string, method: string, confidence: number) => void
  setUmManual: (umPerPixel: number) => void
  setRulerMm: (mm: number) => void
  clear: () => void
}

export const useCalibrationStore = create<CalibrationState>()(
  persist(
    (set) => ({
      umPerPixel: null,
      rulerMm: 10,
      calibrationId: null,
      method: null,
      confidence: null,
      imageUmPerPixel: {},
      setImageCalibration: (imageId, value) => set((state) => {
        const values = { ...state.imageUmPerPixel }
        if (value != null && Number.isFinite(value) && value > 0) values[imageId] = value
        else delete values[imageId]
        return { imageUmPerPixel: values }
      }),
      setCalibration: (umPerPixel, calibrationId, method, confidence) =>
        set({ umPerPixel, calibrationId, method, confidence }),
      setUmManual: (umPerPixel) => set({ umPerPixel }),
      setRulerMm: (mm) => set({ rulerMm: mm }),
      clear: () => set({ umPerPixel: null, calibrationId: null, method: null, confidence: null, imageUmPerPixel: {} }),
    }),
    { name: 'calibration' },
  ),
)
