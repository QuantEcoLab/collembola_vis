import { Routes, Route } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import RequireAuth from './components/auth/RequireAuth'
import LoginPage from './pages/LoginPage'
import CalibratePage from './pages/CalibratePage'
import DetectPage from './pages/DetectPage'
import MeasurePage from './pages/MeasurePage'
import ResultsPage from './pages/ResultsPage'
import JobsPage from './pages/JobsPage'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppShell />}>
          <Route path="/" element={<CalibratePage />} />
          <Route path="/detect" element={<DetectPage />} />
          <Route path="/measure" element={<MeasurePage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/jobs" element={<JobsPage />} />
        </Route>
      </Route>
    </Routes>
  )
}
