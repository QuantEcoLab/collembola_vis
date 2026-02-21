import { NavLink, useNavigate } from 'react-router-dom'
import { Crosshair, Ruler, ScanSearch, Table2, LayoutList, LogOut } from 'lucide-react'
import { useAuthStore } from '../../store/authStore'

const links = [
  { to: '/', label: 'Calibrate', icon: Ruler },
  { to: '/detect', label: 'Detect', icon: ScanSearch },
  { to: '/measure', label: 'Measure', icon: Crosshair },
  { to: '/results', label: 'Results', icon: Table2 },
  { to: '/jobs', label: 'Jobs', icon: LayoutList },
]

export default function Sidebar() {
  const logout = useAuthStore((s) => s.logout)
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <aside className="w-56 shrink-0 bg-gray-900 text-gray-300 flex flex-col">
      <div className="px-4 py-5 border-b border-gray-800">
        <h1 className="text-lg font-bold text-white leading-tight">Collembola</h1>
        <p className="text-xs text-gray-500">Detection Pipeline</p>
      </div>
      <nav className="flex-1 py-3 space-y-0.5">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-gray-800 text-white border-l-2 border-blue-500'
                  : 'hover:bg-gray-800/50 border-l-2 border-transparent'
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-gray-800 p-3">
        <button
          onClick={handleLogout}
          className="flex items-center gap-3 px-4 py-2.5 text-sm w-full rounded hover:bg-gray-800/50 transition-colors"
        >
          <LogOut size={18} />
          Sign out
        </button>
      </div>
    </aside>
  )
}
