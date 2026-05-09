'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import Papa from 'papaparse';

// MapComponent musi być ładowany dynamicznie bez SSR, ponieważ Leaflet wymaga dostępu do obiektu 'window'
const MapComponent = dynamic(() => import('./components/MapComponent'), {
  ssr: false,
  loading: () => <div className="h-[600px] w-full flex items-center justify-center bg-gray-100 font-sans">Ładowanie mapy...</div>
});

interface GPSData {
  signaldate: string;
  LAT: number;
  LON: number;
  // Pola ze zklasyfikowanych danych (opcjonalne)
  outlier?: number;
  inPort?: number;
  atSeaAnchor?: number;
  atSeaAdrift?: number;
  atSeaVoyage?: number;
}

export default function Home() {
  const [data, setData] = useState<GPSData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [allValidData, setAllValidData] = useState<GPSData[]>([]);
  const [currentZoom, setCurrentZoom] = useState<number>(10);

  // Pobieranie listy plików
  useEffect(() => {
    const fetchFileList = async () => {
      try {
        const response = await fetch('/api/files');
        if (response.ok) {
          const fileList = await response.json();
          setFiles(fileList);
          // Jeśli są pliki, wybierz pierwszy domyślnie
          if (fileList.length > 0) {
            setSelectedFile(fileList[0]);
          } else {
            setLoading(false);
          }
        }
      } catch (err) {
        console.error('Błąd podczas pobierania listy plików:', err);
      }
    };
    fetchFileList();
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      if (!selectedFile) return;
      
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`/data/${selectedFile}`);
        
        if (!response.ok) {
          throw new Error('Nie udało się pobrać pliku');
        }

        const csvString = await response.text();
        
        Papa.parse(csvString, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            // Filtrujemy dane, aby upewnić się, że mamy poprawne koordynaty
            const validData = (results.data as any[]).filter(
              (row) => row && row.LAT !== undefined && row.LON !== undefined && row.LAT !== null && row.LON !== null && !isNaN(parseFloat(row.LAT)) && !isNaN(parseFloat(row.LON))
            ).map(row => ({
              signaldate: String(row.signaldate || ''),
              LAT: parseFloat(row.LAT),
              LON: parseFloat(row.LON),
              outlier: row['Outlier GPS'],
              inPort: row['In Port'],
              atSeaAnchor: row['At Sea Anchor'],
              atSeaAdrift: row['At Sea Adrift'] || row['At Sea Adrift GPS'], // Obsługa obu wersji nazwy
              atSeaVoyage: row['At Sea Voyage']
            }));
            
            setAllValidData(validData);
            setLoading(false);
          },
          error: (error: any) => {
            console.error('Błąd podczas parsowania CSV:', error);
            setError('Błąd podczas ładowania danych mapy.');
            setLoading(false);
          }
        });
      } catch (err) {
        console.error('Błąd podczas pobierania pliku:', err);
        setError('Nie udało się pobrać danych GPS.');
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedFile]);

  // Supercluster zajmuje się klastrowaniem - przekazujemy wszystkie dane
  useEffect(() => {
    setData(allValidData);
  }, [allValidData]);

  return (
    <div className="flex flex-col min-h-screen bg-zinc-50 font-sans dark:bg-black">
      <main className="flex flex-1 flex-col p-8 md:p-16">
        <div className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-black dark:text-zinc-50 mb-2">
              Interaktywna Mapa Pozycji Statku
            </h1>
            <p className="text-zinc-600 dark:text-zinc-400">
              Wizualizacja trasy statku na podstawie danych GPS. Wybierz plik danych do wyświetlenia.
            </p>
          </div>

          <div className="flex flex-col gap-2 min-w-[300px]">
            <label htmlFor="file-select" className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Wybierz plik danych CSV:
            </label>
            <select
              id="file-select"
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              className="p-2 border border-zinc-300 rounded-md bg-white dark:bg-zinc-800 dark:border-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-sm focus:ring-2 focus:ring-blue-500 outline-hidden"
            >
              {files.length === 0 && <option value="">Brak dostępnych plików</option>}
              {files.map(file => (
                <option key={file} value={file}>
                  {file}
                </option>
              ))}
            </select>
          </div>

        </div>

        {error ? (
          <div className="p-4 bg-red-100 text-red-700 rounded-lg">{error}</div>
        ) : loading ? (
          <div className="h-[600px] w-full flex items-center justify-center bg-gray-100 font-sans rounded-xl shadow-xl dark:bg-zinc-900 dark:text-zinc-400">
            Wczytywanie danych...
          </div>
        ) : data.length > 0 ? (
          <div className="w-full h-full min-h-[600px] shadow-xl rounded-xl overflow-hidden bg-white">
            <MapComponent data={data} onZoomChange={setCurrentZoom} />
          </div>
        ) : (
          <div className="h-[600px] w-full flex items-center justify-center bg-gray-100 font-sans rounded-xl shadow-xl dark:bg-zinc-900 dark:text-zinc-400">
            Brak danych do wyświetlenia dla wybranego pliku.
          </div>
        )}

        <div className="mt-8 text-sm text-zinc-500 flex justify-between">
          <span>
            Załadowano {allValidData.length} punktów trasy (klastrowanie przy zoomie {currentZoom}).
          </span>
          {selectedFile && <span>Aktualny plik: <code className="bg-zinc-200 dark:bg-zinc-800 px-1 rounded">{selectedFile}</code></span>}
        </div>
      </main>
    </div>
  );
}
