import { create } from 'zustand'
import { persist } from 'zustand/middleware'

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
      logout: () => set({ token: null, role: null, username: null }),
    }),
    { name: 'auth' },
  ),
)
