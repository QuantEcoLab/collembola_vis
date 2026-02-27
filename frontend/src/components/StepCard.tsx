import React from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

interface StepCardProps {
  stepNumber: number;
  title: string;
  status: 'pending' | 'active' | 'complete' | 'loading';
  children: React.ReactNode;
  collapsible?: boolean;
}

export function StepCard({
  stepNumber,
  title,
  status,
  children,
  collapsible = false,
}: StepCardProps) {
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  const statusIcon = {
    pending: <Circle className="w-5 h-5 text-gray-400" />,
    active: <Circle className="w-5 h-5 text-blue-600 fill-blue-600" />,
    complete: <CheckCircle2 className="w-5 h-5 text-green-600" />,
    loading: <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />,
  }[status];

  const statusColors = {
    pending: 'bg-gray-50 border-gray-200',
    active: 'bg-blue-50 border-blue-300',
    complete: 'bg-green-50 border-green-300',
    loading: 'bg-blue-50 border-blue-300',
  }[status];

  const handleToggle = () => {
    if (collapsible) {
      setIsCollapsed(!isCollapsed);
    }
  };

  return (
    <div className={`border rounded-lg ${statusColors} mb-4`}>
      {/* Header */}
      <div
        className={`flex items-center gap-3 p-4 ${
          collapsible ? 'cursor-pointer hover:bg-black/5' : ''
        }`}
        onClick={handleToggle}
      >
        {statusIcon}
        <div className="flex-1">
          <div className="text-sm font-medium text-gray-900">
            Step {stepNumber}: {title}
          </div>
        </div>
        {collapsible && (
          <button className="text-gray-500 hover:text-gray-700">
            <svg
              className={`w-5 h-5 transition-transform ${
                isCollapsed ? '' : 'rotate-180'
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>
        )}
      </div>

      {/* Body */}
      {!isCollapsed && (
        <div className="px-4 pb-4 space-y-3">
          {children}
        </div>
      )}
    </div>
  );
}
