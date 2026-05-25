import { useEffect, useRef, useState, useCallback } from 'react';
import { useDispatch } from 'react-redux';
import { KeplerGl } from '@kepler.gl/components';
import { addDataToMap } from '@kepler.gl/actions';
import { processCsvData } from '@kepler.gl/processors';

const MAPBOX_TOKEN = import.meta.env.MAP_TOKEN;

const MAP_CONFIG = {
  visState: {
    layers: [
      {
        id: 'ship-layer',
        type: 'point',
        config: {
          dataId: 'ship_data',
          label: 'Operacje statku',
          columns: {
            lat: 'LAT',
            lng: 'LON',
            altitude: null
          },
          isVisible: true,
          color: [255, 255, 255],
          colorField: {
            name: 'status_id',
            type: 'integer'
          },
          colorScale: 'ordinal',
          visConfig: {
            radius: 6,
            fixedRadius: false,
            opacity: 0.9,
            outline: false,
            thickness: 2,
            strokeColor: null,
            colorRange: {
              name: "color.customPalette.customOrdinal.status_id",
              type: "customOrdinal",
              category: "Custom",
              colors: [
                "#e41a1c",
                "#984ea3",
                "#377eb8",
                "#4daf4a",
                "#E48928",
                "#000000"
              ],
              colorMap: [
                [
                  1,
                  "#e41a1c"
                ],
                [
                  2,
                  "#984ea3"
                ],
                [
                  3,
                  "#377eb8"
                ],
                [
                  [
                    4
                  ],
                  "#4daf4a"
                ],
                [
                  [
                    5
                  ],
                  "#E48928"
                ],
                [
                  [
                    6
                  ],
                  "#000000"
                ]
              ]
            },
            strokeColorRange: {
              name: "Global Warming",
              type: "sequential",
              category: "Uber",
              colors: [
                "#4C0035",
                "#880030",
                "#B72F15",
                "#D6610A",
                "#EF9100",
                "#FFC300"
              ]
            },
            radiusRange: [
              0,
              50
            ],
            filled: true,
            billboard: false,
            allowHover: true,
            showNeighborOnHover: false,
            showHighlightColor: true
          },
        },
        visualChannels: {
          colorField: {
            name: "status_id",
            type: "integer"
          },
          colorScale: "customOrdinal",
          strokeColorField: null,
          strokeColorScale: "quantile",
          sizeField: null,
          sizeScale: "linear"
        }
      }
    ],
    filters: [],
    interactionConfig: {
      tooltip: {
        fieldsToShow: {
          ship_data: [
            { name: 'signaldate', format: null },
            { name: 'Outlier GPS', format: null },
            { name: 'In Port', format: null },
            { name: 'At Sea Anchor', format: null },
            { name: 'At Sea Adrift', format: null },
            { name: 'At Sea Adrift GPS', format: null },
            { name: 'At Sea Voyage', format: null },
            { name: 'LAT', format: null },
            { name: 'LON', format: null },
            { name: 'status_id', format: null },
          ]
        },
        compareMode: false,
        compareType: 'absolute',
        enabled: true
      }
    }
  },
  mapState: {
    bearing: 0,
    dragRotate: false,
    latitude: 44.1,
    longitude: 28.66,
    pitch: 0,
    zoom: 10,
    isSplit: false
  },
  mapStyle: {
    styleType: 'dark'
  }
};

const overlayStyle = {
  position: 'absolute',
  top: '16px',
  left: '50%',
  transform: 'translateX(-50%)',
  zIndex: 9999,
  display: 'flex',
  alignItems: 'center',
  gap: '10px',
  background: 'rgba(30,30,40,0.92)',
  border: '1px solid #444',
  borderRadius: '8px',
  padding: '10px 18px',
  boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
  color: '#fff',
  fontFamily: 'system-ui, sans-serif',
  fontSize: '14px'
};

const btnStyle = {
  background: '#3a86ff',
  color: '#fff',
  border: 'none',
  borderRadius: '6px',
  padding: '7px 16px',
  cursor: 'pointer',
  fontSize: '14px',
  fontWeight: 600,
  whiteSpace: 'nowrap'
};

function App() {
  const dispatch = useDispatch();
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const datasetCounter = useRef(0);

  const loadCsv = useCallback((csvText, name) => {
    try {
      const data = processCsvData(csvText);
      datasetCounter.current += 1;
      const dataId = `ship_data_${datasetCounter.current}`;

      const { fields, rows } = data;

      // 1. Wykrywanie kolumn (szukanie słów kluczowych)
      const f = (term) => fields.findIndex(f => f.name.toLowerCase().includes(term.toLowerCase()));
      const iOut = f('outlier'), iPort = f('in port'), iVoy = f('voyage'), iAnc = f('anchor'), iAdr = f('adrift');

      // 2. Dodajemy techniczną kolumnę status_id (integer)
      fields.push({ name: 'status_id', type: 'integer' });

      rows.forEach(row => {
        let id;

        if (iOut !== -1 && row[iOut] === 1)      id = 1; // Outlier
        else if (iPort !== -1 && row[iPort] === 1) id = 2; // In Port
        else if (iVoy !== -1 && row[iVoy] === 1)  id = 3; // Sea Voyage
        else if (iAnc !== -1 && row[iAnc] === 1)  id = 4; // Sea Anchor
        else if (iAdr !== -1 && row[iAdr] === 1)  id = 5; // Sea Adrift
        else                                      id = 6; // Wszystko równe 0 lub brak dopasowania

        row.push(id);
      });

      const layerConfig = {
        ...MAP_CONFIG,
        visState: {
          ...MAP_CONFIG.visState,
          layers: [
            {
              ...MAP_CONFIG.visState.layers[0],
              id: `ship-layer-${datasetCounter.current}`,
              config: {
                ...MAP_CONFIG.visState.layers[0].config,
                dataId,
                label: name || `Dane CSV ${datasetCounter.current}`
              }
            }
          ],
          interactionConfig: {
            ...MAP_CONFIG.visState.interactionConfig,
            tooltip: {
              ...MAP_CONFIG.visState.interactionConfig.tooltip,
              fieldsToShow: {
                ...MAP_CONFIG.visState.interactionConfig.tooltip.fieldsToShow,
                [dataId]: MAP_CONFIG.visState.interactionConfig.tooltip.fieldsToShow['ship_data']
              }
            }
          }
        }
      };
      dispatch(
          addDataToMap({
            datasets: {
              info: {
                label: name || `Dane CSV ${datasetCounter.current}`,
                id: dataId
              },
              data
            },
            options: {
              centerMap: true,
              readOnly: false,
              keepExistingConfig: true
            },
            config: layerConfig
          })
      );
      setFiles(prev => [...prev, name || `Dane CSV ${datasetCounter.current}`]);
      setError(null);
    } catch (e) {
      setError('Błąd wczytywania pliku: ' + e.message);
    }
  }, [dispatch]);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.name.endsWith('.csv')) {
      setError('Wybierz plik CSV.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => loadCsv(e.target.result, file.name);
    reader.readAsText(file);
  }, [loadCsv]);

  useEffect(() => {
    const blockDrop = (e) => { e.preventDefault(); e.stopPropagation(); };
    const handleWindowDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const file = e.dataTransfer?.files?.[0];
      if (file) handleFile(file);
      setDragging(false);
    };
    window.addEventListener('dragover', blockDrop, true);
    window.addEventListener('drop', handleWindowDrop, true);
    return () => {
      window.removeEventListener('dragover', blockDrop, true);
      window.removeEventListener('drop', handleWindowDrop, true);
    };
  }, [handleFile]);

  const handleInputChange = (e) => handleFile(e.target.files[0]);

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setDragging(true); };
  const handleDragLeave = () => setDragging(false);

  return (
      <div
          style={{ position: 'absolute', inset: 0 }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
      >
        <div style={{ ...overlayStyle, outline: dragging ? '2px solid #3a86ff' : 'none' }}>
          <span>{files.length > 0 ? `📄 Załadowano: ${files.length} plik(ów)` : 'Załaduj plik CSV z danymi statku'}</span>
          <button style={btnStyle} onClick={() => fileInputRef.current.click()}>
            Dodaj plik
          </button>
          <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleInputChange}
          />
          {error && <span style={{ color: '#ff6b6b', marginLeft: 8 }}>{error}</span>}
          {files.length === 0 && <span style={{ color: '#aaa', fontSize: 12 }}>lub przeciągnij plik tutaj</span>}
        </div>
        <KeplerGl
            id="ship_map"
            mapboxApiAccessToken={MAPBOX_TOKEN}
            width={window.innerWidth}
            height={window.innerHeight}
        />
      </div>
  );
}

export default App;
