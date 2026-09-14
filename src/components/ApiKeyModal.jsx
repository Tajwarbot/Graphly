import { useState, useEffect, useRef } from 'react';
import { X, ExternalLink, Eye, EyeOff } from 'lucide-react';
import { useModalFocus } from './useModalFocus';
import './SetupUI.css';
import { getGeminiApiKey, setGeminiApiKey, removeGeminiApiKey } from '../lib/security';

export function ApiKeyModal({ isOpen, onClose, onSave }) {
    return isOpen ? <ApiKeyForm onClose={onClose} onSave={onSave} /> : null;
}

function ApiKeyForm({ onClose, onSave }) {
    const [apiKey, setApiKey] = useState(() => getGeminiApiKey() || '');
    const [hasSavedKey, setHasSavedKey] = useState(() => Boolean(getGeminiApiKey()));
    const timer = useRef(null);
    const [showKey, setShowKey] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const dialog = useRef(null);
    useModalFocus(dialog, onClose);

    useEffect(() => () => clearTimeout(timer.current), []);

    const handleSave = () => {
        if (!apiKey.trim()) {
            setError('Please enter a valid API key');
            return;
        }

        if (apiKey.length < 30) {
            setError('API key appears to be too short. Please check again.');
            return;
        }

        try {
            setGeminiApiKey(apiKey.trim());
            window.dispatchEvent(new Event('graphly-key-change'));
        } catch {
            setError('Your browser could not save the key. Check your storage settings and try again.');
            return;
        }
        setHasSavedKey(true);
        setSuccess('API key saved successfully!');
        setError('');

        clearTimeout(timer.current);
        timer.current = setTimeout(() => {
            onSave();
            onClose();
        }, 800);
    };

    const handleRemove = () => {
        if (confirm('Are you sure you want to remove your API key? You will need to enter it again to use AI features.')) {
            try {
                removeGeminiApiKey();
                window.dispatchEvent(new Event('graphly-key-change'));
            } catch {
                setError('Your browser could not remove the key. Please try again.');
                return;
            }
            setHasSavedKey(false);
            setApiKey('');
            setSuccess('API key removed.');
            clearTimeout(timer.current);
            timer.current = setTimeout(() => {
                setSuccess('');
            }, 1500);
        }
    };

    return (
        <div className="graphly-modal-backdrop">
            <section ref={dialog} className="graphly-modal" role="dialog" aria-modal="true" aria-labelledby="api-key-title">
                <header className="graphly-modal-header">
                    <h2 id="api-key-title">Connect AI</h2>
                    <button className="graphly-icon-button" onClick={onClose} aria-label="Close AI settings"><X size={20} /></button>
                </header>
                <div className="graphly-modal-body">
                    <div className="graphly-provider"><strong>Google Gemini</strong><span>AI provider</span></div>
                    <p>Use your own key for the graph assistant and image scanning. Graphing and CSV import work without a key.</p>
                    <div className="graphly-key-help">
                        <p>Get a key in Google AI Studio, then paste it below.</p>
                        <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer">Get a Gemini API key <ExternalLink size={14} aria-hidden="true" /></a>
                    </div>
                    <div>
                        <label htmlFor="gemini-api-key">Gemini API key</label>
                        <div className="graphly-key-input">
                            <input id="gemini-api-key" className="graphly-modal-input" type={showKey ? 'text' : 'password'} autoComplete="off" spellCheck={false} value={apiKey} onChange={event => { setApiKey(event.target.value); setError(''); }} placeholder="Paste your API key" aria-describedby="key-storage-note" />
                            <button type="button" className="graphly-icon-button" onClick={() => setShowKey(!showKey)} aria-label={showKey ? 'Hide API key' : 'Show API key'}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button>
                        </div>
                    </div>
                    <p id="key-storage-note">Saved in this browser. AI requests send your prompts and any attached files to Google.</p>
                    {error && <p role="alert" className="graphly-setup-error">{error}</p>}
                    {success && <p role="status" className="graphly-setup-success">{success}</p>}
                </div>
                <footer className="graphly-modal-footer">
                    {hasSavedKey && <button className="graphly-button graphly-button-danger" onClick={handleRemove}>Remove key</button>}
                    <button className="graphly-button" onClick={onClose}>Cancel</button>
                    <button className="graphly-button graphly-button-primary" disabled={!apiKey.trim()} onClick={handleSave}>Save key</button>
                </footer>
            </section>
        </div>
    );
}

export default ApiKeyModal;
