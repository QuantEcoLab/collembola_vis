import { Routes, Route } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import CalibratePage from './pages/CalibratePage'
import DetectPage from './pages/DetectPage'
import MeasurePage from './pages/MeasurePage'
import ResultsPage from './pages/ResultsPage'
import JobsPage from './pages/JobsPage'

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<CalibratePage />} />
        <Route path="/detect" element={<DetectPage />} />
        <Route path="/measure" element={<MeasurePage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/jobs" element={<JobsPage />} />
      </Route>
    </Routes>
  )
}
