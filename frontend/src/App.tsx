import { Routes, Route } from 'react-router-dom'
import RequireAuth from './components/auth/RequireAuth'
import LoginPage from './pages/LoginPage'
import WorkspacePage from './pages/WorkspacePage'
import ProjectsPage from './pages/ProjectsPage'
import ProjectDetailPage from './pages/ProjectDetailPage'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:id" element={<ProjectDetailPage />} />
        <Route path="/*" element={<WorkspacePage />} />
      </Route>
    </Routes>
  )
}
