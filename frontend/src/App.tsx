import { Routes, Route } from 'react-router-dom'
import RequireAuth from './components/auth/RequireAuth'
import LoginPage from './pages/LoginPage'
import WorkspacePage from './pages/WorkspacePage'
import CollaboratePage from './pages/CollaboratePage'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route path="/collaborate" element={<CollaboratePage />} />
        <Route path="/*" element={<WorkspacePage />} />
      </Route>
    </Routes>
  )
}
