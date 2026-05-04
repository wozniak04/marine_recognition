'use client';

import { MapContainer, TileLayer, CircleMarker, Popup, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import L from 'leaflet';
import Supercluster from 'supercluster';

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
  onZoomChange?: (zoom: number) => void;
}

type PointFeature = GeoJSON.Feature<GeoJSON.Point, GPSData>;

function getPointStyle(point: GPSData) {
  if (point.outlier === 1) return { color: '#ef4444', label: 'Outlier' };
  if (point.inPort === 1) return { color: '#8b5cf6', label: 'W porcie' };
  if (point.atSeaAnchor === 1) return { color: '#f59e0b', label: 'Na kotwicy' };
  if (point.atSeaAdrift === 1) return { color: '#10b981', label: 'Dryfowanie' };
  if (point.atSeaVoyage === 1) return { color: '#2563eb', label: 'W podróży' };
  return { color: '#2563eb', label: 'Pozycja GPS' };
}

function getClusterBreakdown(index: Supercluster, clusterId: number): Record<string, number> {
  const leaves = index.getLeaves(clusterId, Infinity);
  const breakdown: Record<string, number> = {};
  for (const leaf of leaves) {
    const p = leaf.properties as GPSData;
    const { label } = getPointStyle(p);
    breakdown[label] = (breakdown[label] || 0) + 1;
  }
  return breakdown;
}

const LABEL_COLORS: Record<string, string> = {
  'Outlier': '#ef4444',
  'W porcie': '#8b5cf6',
  'Na kotwicy': '#f59e0b',
  'Dryfowanie': '#10b981',
  'W podróży': '#2563eb',
  'Pozycja GPS': '#2563eb',
};

function hexToRgb(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b];
}

function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
}

function getClusterColor(index: Supercluster, clusterId: number): { fill: string; border: string } {
  const breakdown = getClusterBreakdown(index, clusterId);
  const total = Object.values(breakdown).reduce((a, b) => a + b, 0);
  let r = 0, g = 0, b = 0;
  for (const [label, cnt] of Object.entries(breakdown)) {
    const weight = cnt / total;
    const color = LABEL_COLORS[label] || '#2563eb';
    const [cr, cg, cb] = hexToRgb(color);
    r += cr * weight;
    g += cg * weight;
    b += cb * weight;
  }
  const fill = rgbToHex(r, g, b);
  // Darker border
  const border = rgbToHex(r * 0.7, g * 0.7, b * 0.7);
  return { fill, border };
}

function ClusterPopupContent({ superclusterIndex, clusterId, count }: { superclusterIndex: Supercluster; clusterId: number; count: number }) {
  const breakdown = useMemo(() => getClusterBreakdown(superclusterIndex, clusterId), [superclusterIndex, clusterId]);

  return (
    <div className="text-xs font-semibold text-center min-w-[120px]">
      <div className="mb-1">{count} punktów</div>
      <div className="flex flex-col gap-0.5 text-left">
        {Object.entries(breakdown)
          .sort((a, b) => b[1] - a[1])
          .map(([label, cnt]) => (
            <div key={label} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0 inline-block"
                style={{ backgroundColor: LABEL_COLORS[label] || '#2563eb' }}
              />
              <span className="text-gray-700 font-normal">{label}:</span>
              <span className="font-semibold">{cnt}</span>
            </div>
          ))}
      </div>
      <div className="text-gray-500 mt-1">Kliknij aby przybliżyć</div>
    </div>
  );
}

function ClusterLayer({ data, onZoomChange }: { data: GPSData[]; onZoomChange?: (zoom: number) => void }) {
  const map = useMap();
  const [clusters, setClusters] = useState<any[]>([]);
  const indexRef = useRef<Supercluster | null>(null);

  // Budowanie indeksu supercluster
  const superclusterIndex = useMemo(() => {
    const points: Supercluster.PointFeature<GPSData>[] = data
      .filter(p => p && !isNaN(p.LAT) && !isNaN(p.LON))
      .map((point, i) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Point' as const,
          coordinates: [point.LON, point.LAT],
        },
        properties: point,
      }));

    const index = new Supercluster({
      radius: 50,
      maxZoom: 16,
      minZoom: 0,
    });
    index.load(points);
    return index;
  }, [data]);

  const updateClusters = useCallback(() => {
    const bounds = map.getBounds();
    const zoom = map.getZoom();

    if (onZoomChange) {
      onZoomChange(zoom);
    }

    const bbox: GeoJSON.BBox = [
      bounds.getWest(),
      bounds.getSouth(),
      bounds.getEast(),
      bounds.getNorth(),
    ];

    const visibleClusters = superclusterIndex.getClusters(
      bbox as [number, number, number, number],
      Math.floor(zoom)
    );
    setClusters(visibleClusters);
  }, [map, superclusterIndex, onZoomChange]);

  useMapEvents({
    zoomend: updateClusters,
    moveend: updateClusters,
  });

  // Inicjalne załadowanie klastrów
  useEffect(() => {
    updateClusters();
  }, [updateClusters]);

  indexRef.current = superclusterIndex;

  return (
    <>
      {clusters.map((cluster, idx) => {
        const [lng, lat] = cluster.geometry.coordinates;
        const isCluster = cluster.properties.cluster;

        if (isCluster) {
          const count = cluster.properties.point_count;
          const size = Math.min(40, 12 + Math.sqrt(count) * 2);

          const clusterColors = getClusterColor(superclusterIndex, cluster.id as number);

          return (
            <CircleMarker
              key={`cluster-${cluster.id}`}
              center={[lat, lng]}
              radius={size}
              fillColor={clusterColors.fill}
              color={clusterColors.border}
              weight={2}
              opacity={0.9}
              fillOpacity={0.7}
              eventHandlers={{
                click: () => {
                  const expansionZoom = Math.min(
                    superclusterIndex.getClusterExpansionZoom(cluster.id as number),
                    18
                  );
                  map.flyTo([lat, lng], expansionZoom, { duration: 0.5 });
                },
              }}
            >
              <Popup>
                <ClusterPopupContent
                  superclusterIndex={superclusterIndex}
                  clusterId={cluster.id as number}
                  count={count}
                />
              </Popup>
            </CircleMarker>
          );
        }

        // Pojedynczy punkt
        const point = cluster.properties as GPSData;
        const style = getPointStyle(point);

        return (
          <CircleMarker
            key={`point-${idx}-${point.signaldate}`}
            center={[lat, lng]}
            radius={4}
            fillColor={style.color}
            color={style.color}
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
    </>
  );
}

export default function MapComponent({ data, onZoomChange }: MapComponentProps) {
  const [isClient, setIsClient] = useState(false);

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
          <div className="flex items-center gap-3 text-gray-700">
            <span className="w-5 h-5 rounded-full bg-[#3b82f6] border-2 border-[#1d4ed8] shadow-sm shrink-0 flex items-center justify-center text-[8px] text-white font-bold">N</span>
            <span>Klaster (N punktów)</span>
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
        <ClusterLayer data={data} onZoomChange={onZoomChange} />
      </MapContainer>
    </div>
  );
}
