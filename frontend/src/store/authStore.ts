import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AuthState {
  token: string | null
  role: string | null
  login: (token: string, role: string) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      role: null,
      login: (token, role) => set({ token, role }),
      logout: () => set({ token: null, role: null }),
    }),
    { name: 'auth' },
  ),
)
