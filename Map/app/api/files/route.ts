import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const dataDirectory = path.join(process.cwd(), 'public/data');
    
    // Upewnij się, że katalog istnieje
    if (!fs.existsSync(dataDirectory)) {
      return NextResponse.json([]);
    }

    const files = fs.readdirSync(dataDirectory);
    
    // Filtruj tylko pliki CSV
    const csvFiles = files.filter(file => file.toLowerCase().endsWith('.csv'));
    
    return NextResponse.json(csvFiles);
  } catch (error) {
    console.error('Błąd podczas listowania plików:', error);
    return NextResponse.json({ error: 'Nie udało się pobrać listy plików' }, { status: 500 });
  }
}
