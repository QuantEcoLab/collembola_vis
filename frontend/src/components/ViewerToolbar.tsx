import React from 'react';
import { Image, Download, Eye } from 'lucide-react';

export type OverlayMode = 'raw' | 'boxes' | 'contours' | 'both';

interface ViewerToolbarProps {
  overlayMode: OverlayMode;
  onOverlayChange: (mode: OverlayMode) => void;
  availableOverlays: {
    boxes: boolean;
    contours: boolean;
  };
  onExport?: (format: 'image' | 'csv' | 'excel') => void;
  measurementDone?: boolean;
}

export function ViewerToolbar({
  overlayMode,
  onOverlayChange,
  availableOverlays,
  onExport,
  measurementDone = false,
}: ViewerToolbarProps) {
  const [exportMenuOpen, setExportMenuOpen] = React.useState(false);

  const overlayOptions: { value: OverlayMode; label: string; disabled: boolean }[] = [
    { value: 'raw', label: 'Raw Image', disabled: false },
    { value: 'boxes', label: 'Detection Boxes', disabled: !availableOverlays.boxes },
    { value: 'contours', label: 'SAM Contours', disabled: !availableOverlays.contours },
    { value: 'both', label: 'Boxes + Contours', disabled: !availableOverlays.boxes || !availableOverlays.contours },
  ];

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
      {/* Left side: Overlay selector */}
      <div className="flex items-center gap-3">
        <Eye className="w-4 h-4 text-gray-600" />
        <span className="text-sm font-medium text-gray-700">View:</span>
        <select
          value={overlayMode}
          onChange={(e) => onOverlayChange(e.target.value as OverlayMode)}
          className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm bg-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {overlayOptions.map((option) => (
            <option key={option.value} value={option.value} disabled={option.disabled}>
              {option.label}
              {option.disabled && ' (unavailable)'}
            </option>
          ))}
        </select>
      </div>

      {/* Right side: Export menu */}
      {onExport && (
        <div className="relative">
          <button
            onClick={() => setExportMenuOpen(!exportMenuOpen)}
            className="flex items-center gap-2 px-3 py-1.5 border border-gray-300 rounded-lg text-sm bg-white hover:bg-gray-50 font-medium transition-colors"
          >
            <Download className="w-4 h-4" />
            Export
          </button>

          {/* Export dropdown menu */}
          {exportMenuOpen && (
            <>
              {/* Backdrop to close menu */}
              <div
                className="fixed inset-0 z-10"
                onClick={() => setExportMenuOpen(false)}
              />

              {/* Menu */}
              <div className="absolute right-0 mt-1 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-20">
                <button
                  onClick={() => {
                    onExport('image');
                    setExportMenuOpen(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
                >
                  <Image className="w-4 h-4" />
                  Download Image
                </button>
                <button
                  onClick={() => {
                    onExport('csv');
                    setExportMenuOpen(false);
                  }}
                  disabled={!measurementDone}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 disabled:text-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download CSV
                </button>
                <button
                  onClick={() => {
                    onExport('excel');
                    setExportMenuOpen(false);
                  }}
                  disabled={!measurementDone}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 disabled:text-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Excel
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
