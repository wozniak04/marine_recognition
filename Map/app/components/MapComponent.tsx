'use client';

import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useEffect, useState } from 'react';
import L from 'leaflet';

interface GPSData {
  signaldate: string;
  LAT: number;
  LON: number;
  outlier?: number;
  inPort?: number;
  atSeaAnchor?: number;
  atSeaAdrift?: number;
  atSeaVoyage?: number;
}

interface MapComponentProps {
  data: GPSData[];
}

export default function MapComponent({ data }: MapComponentProps) {
  const [isClient, setIsClient] = useState(false);

  // Naprawa ikon Leaflet w Next.js
  useEffect(() => {
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
      iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
      shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
    });
    setIsClient(true);
  }, []);

  if (!isClient || data.length === 0 || !data[0]) {
    return <div className="h-[600px] w-full flex items-center justify-center bg-gray-100">Ładowanie mapy...</div>;
  }

  const center: [number, number] = [data[0].LAT, data[0].LON];

  const getPointStyle = (point: GPSData) => {
    if (point.outlier === 1) return { color: '#ef4444', label: 'Outlier' }; // Czerwony
    if (point.inPort === 1) return { color: '#8b5cf6', label: 'W porcie' }; // Fioletowy
    if (point.atSeaAnchor === 1) return { color: '#f59e0b', label: 'Na kotwicy' }; // Pomarańczowy
    if (point.atSeaAdrift === 1) return { color: '#10b981', label: 'Dryfowanie' }; // Zielony
    if (point.atSeaVoyage === 1) return { color: '#2563eb', label: 'W podróży' }; // Niebieski
    return { color: '#2563eb', label: 'Pozycja GPS' }; // Domyślny niebieski
  };

  return (
    <div className="h-[600px] w-full rounded-lg overflow-hidden border border-gray-300 relative">
      <div className="absolute top-4 right-4 z-[1000] bg-white p-4 rounded-xl shadow-2xl border border-gray-200 text-sm font-sans min-w-[150px] backdrop-blur-md bg-white/95">
        <h4 className="font-bold mb-3 text-gray-800 border-b pb-2 border-gray-100">Legenda</h4>
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center gap-3 text-gray-700">
            <span className="w-4 h-4 rounded-full bg-[#2563eb] border border-white shadow-sm shrink-0"></span> 
            <span>W podróży</span>
          </div>
          <div className="flex items-center gap-3 text-gray-700">
            <span className="w-4 h-4 rounded-full bg-[#8b5cf6] border border-white shadow-sm shrink-0"></span> 
            <span>W porcie</span>
          </div>
          <div className="flex items-center gap-3 text-gray-700">
            <span className="w-4 h-4 rounded-full bg-[#f59e0b] border border-white shadow-sm shrink-0"></span> 
            <span>Na kotwicy</span>
          </div>
          <div className="flex items-center gap-3 text-gray-700">
            <span className="w-4 h-4 rounded-full bg-[#10b981] border border-white shadow-sm shrink-0"></span> 
            <span>Dryfowanie</span>
          </div>
          <div className="flex items-center gap-3 text-gray-700">
            <span className="w-4 h-4 rounded-full bg-[#ef4444] border border-white shadow-sm shrink-0"></span> 
            <span>Outlier</span>
          </div>
        </div>
      </div>
      <MapContainer 
        center={center} 
        zoom={10} 
        style={{ height: '100%', width: '100%' }} 
        scrollWheelZoom={true}
        preferCanvas={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        {data.filter(p => p && !isNaN(p.LAT) && !isNaN(p.LON)).map((point, index) => {
          const style = getPointStyle(point);
          return (
            <CircleMarker
              key={`${point.signaldate}-${index}`}
              center={[point.LAT, point.LON]}
              radius={4}
              fillColor={style.color}
              color="white"
              weight={1}
              opacity={1}
              fillOpacity={0.8}
            >
              <Popup>
                <div className="text-xs">
                  <strong>Data:</strong> {point.signaldate}<br />
                  <strong>Status:</strong> {style.label}<br />
                  <strong>Lat:</strong> {point.LAT.toFixed(6)}<br />
                  <strong>Lon:</strong> {point.LON.toFixed(6)}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
