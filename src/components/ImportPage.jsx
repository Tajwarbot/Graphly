import { useRef, useState } from 'react';
import { ArrowLeft, KeyRound, X } from 'lucide-react';
import { useModalFocus } from './useModalFocus';
import './SetupUI.css';

export function ImportPage({ isImporting, image, scanStatus, onImageChange, onScan, onBack, onApiKey, onCSVImport }) {
    const [showCSV, setShowCSV] = useState(false);
    const [fileError, setFileError] = useState('');
    const imageInput = useRef(null);
    const loadImage = event => {
        const file = event.target.files?.[0];
        if (!file) return;
        if (!['image/png', 'image/jpeg', 'image/webp', 'image/gif'].includes(file.type)) { setFileError('Choose a PNG, JPEG, WebP, or GIF image.'); return; }
        if (file.size > 10 * 1024 * 1024) { setFileError('Choose an image smaller than 10 MB.'); return; }
        setFileError('');
        const reader = new FileReader();
        reader.onerror = () => setFileError('Could not read this image. Please try again.');
        reader.onload = () => onImageChange({ file, base64: reader.result.split(',')[1], preview: reader.result });
        reader.readAsDataURL(file);
        event.target.value = '';
    };
    return <main className="graphly-setup">
        <button className="graphly-button" onClick={onBack}><ArrowLeft size={16} aria-hidden="true" />Back to {isImporting ? 'graph' : 'home'}</button>
        <header className="graphly-setup-heading">
            <div><h1>Bring your data.</h1><p>{isImporting ? 'Add data to your current graph.' : 'Start a graph from a table or an image.'}</p></div>
            <button className="graphly-button" onClick={onApiKey}><KeyRound size={16} aria-hidden="true" />AI settings</button>
        </header>
        <div className="graphly-setup-actions">
            <button className="graphly-import-card" onClick={() => imageInput.current?.click()} disabled={scanStatus === 'scanning'}>
                
                {image && <img src={image.preview} alt="Selected table to scan" />}
                <span><strong>{image ? 'Change image' : 'Scan a table'}</strong><small>Choose an image · Gemini key required</small></span>
            </button>
            <button className="graphly-import-card" onClick={() => setShowCSV(true)}>
                
                <span><strong>Import CSV</strong><small>Upload a file or paste data · No key needed</small></span>
            </button>
        </div>
        <input ref={imageInput} type="file" accept="image/png,image/jpeg,image/webp,image/gif" hidden onChange={loadImage} />
        {fileError && <p role="alert" className="graphly-setup-error">{fileError}</p>}
        {image && <button className="graphly-button graphly-button-primary" onClick={onScan} disabled={scanStatus === 'scanning'}>{scanStatus === 'scanning' ? 'Reading your table…' : 'Scan image with Gemini'}</button>}
        {showCSV && <CSVImportDialog onClose={() => setShowCSV(false)} onImport={onCSVImport} />}
    </main>;
}

function CSVImportDialog({ onClose, onImport }) {
    const [text, setText] = useState('');
    const [error, setError] = useState('');
    const dialog = useRef(null);
    useModalFocus(dialog, onClose);
    const importText = () => {
        if (!text.trim()) { setError('Paste some CSV data or choose a file first.'); return; }
        try { onImport(text); } catch (cause) { setError(cause.message || 'Could not import this CSV. Check the data and try again.'); }
    };
    return <div className="graphly-modal-backdrop"><section ref={dialog} className="graphly-modal" role="dialog" aria-modal="true" aria-labelledby="csv-title">
        <header className="graphly-modal-header"><h2 id="csv-title">Import CSV</h2><button className="graphly-icon-button" onClick={onClose} aria-label="Close CSV import"><X size={20} /></button></header>
        <div className="graphly-modal-body">
            <div><label htmlFor="csv-data">Paste your data</label><textarea id="csv-data" className="graphly-modal-input" style={{ minHeight: 160 }} value={text} onChange={event => { setText(event.target.value); setError(''); }} placeholder={'x,y\n1,2\n3,4'} spellCheck={false} /></div>
            <div className="graphly-csv-upload"><label htmlFor="csv-file">Or choose a CSV file</label><input id="csv-file" type="file" accept=".csv,text/csv" onChange={event => {
                const file = event.target.files?.[0];
                if (!file) return;
                if (file.size > 10 * 1024 * 1024) { setError('Choose a CSV smaller than 10 MB.'); return; }
                const reader = new FileReader();
                reader.onerror = () => setError('Could not read this file. Please try again.');
                reader.onload = () => { setText(String(reader.result)); setError(''); };
                reader.readAsText(file);
                event.target.value = '';
            }} /></div>
            {error && <p role="alert" className="graphly-setup-error">{error}</p>}
        </div>
        <footer className="graphly-modal-footer"><button className="graphly-button" onClick={onClose}>Cancel</button><button className="graphly-button graphly-button-primary" onClick={importText}>Import data</button></footer>
    </section></div>;
}
