import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { useProjectStore } from './projectStore'

interface AuthState {
  token: string | null
  role: string | null
  username: string | null
  login: (token: string, role: string, username?: string) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      role: null,
      username: null,
      login: (token, role, username) => set({ token, role, username: username ?? null }),
      logout: () => {
        useProjectStore.getState().setCurrentProject(null, null)
        set({ token: null, role: null, username: null })
      },
    }),
    { name: 'auth' },
  ),
)
