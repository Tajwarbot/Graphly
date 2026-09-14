import { useState, useEffect, useRef } from 'react';
import { X, Key, Save, ExternalLink, Check, AlertCircle, Eye, EyeOff, Trash2, HelpCircle, ChevronRight, ChevronDown } from 'lucide-react';
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
    const [showTutorial, setShowTutorial] = useState(false);

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
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60">
            <div
                role="dialog"
                aria-modal="true"
                aria-labelledby="api-key-title"
                className="bg-white border border-neutral-200 rounded-2xl w-full max-w-lg max-h-[90dvh] overflow-y-auto shadow-xl"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="px-5 py-3 bg-black text-white border-b-2 border-black flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Key size={18} className="text-white" />
                        <h2 id="api-key-title" className="text-base font-bold tracking-tight">API Key Configuration</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 hover:bg-white hover:text-black transition-none text-white border border-transparent hover:border-white"
                        aria-label="Close"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-5">
                    <div className="border border-black p-4 text-sm text-black leading-relaxed">
                        <p>
                            To process images and extract data tables, Graphly connects directly to Google Gemini AI from your browser.
                            Enter your API key below.
                        </p>
                    </div>

                    {/* Input Field */}
                    <div className="space-y-2">
                        <label htmlFor="gemini-api-key" className="text-xs font-bold text-black block">
                            Gemini API Key
                        </label>
                        <div className="relative">
                            <input
                                id="gemini-api-key"
                                autoComplete="off"
                                type={showKey ? "text" : "password"}
                                value={apiKey}
                                onChange={(e) => {
                                    setApiKey(e.target.value);
                                    setError('');
                                }}
                                placeholder="AIzaSy..."
                                className={`
                                    w-full pl-3 pr-10 py-2.5 bg-white border-2 border-black outline-none
                                    text-black font-mono text-sm placeholder:text-neutral-400
                                    ${error ? 'border-red-600' : 'border-black'}
                                `}
                            />
                            <button
                                type="button"
                                onClick={() => setShowKey(!showKey)}
                                className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-black hover:bg-black hover:text-white transition-none"
                                aria-label={showKey ? "Hide key" : "Show key"}
                            >
                                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                            </button>
                        </div>

                        {/* Error / Success Messages */}
                        {error && (
                            <div className="flex items-center gap-2 text-red-600 text-xs font-mono mt-2">
                                <AlertCircle size={14} />
                                <span>{error}</span>
                            </div>
                        )}
                        {success && (
                            <div className="flex items-center gap-2 text-black font-mono text-xs mt-2 bg-neutral-100 p-1.5 border border-black">
                                <Check size={14} />
                                <span>{success}</span>
                            </div>
                        )}
                    </div>

                    {/* Tutorial Accordion */}
                    <div className="border border-black">
                        <button
                            onClick={() => setShowTutorial(!showTutorial)}
                            className="w-full flex items-center justify-between p-3 bg-white hover:bg-black hover:text-white transition-none text-left"
                        >
                            <span className="text-xs font-bold flex items-center gap-2">
                                <HelpCircle size={14} />
                                How to get a free API Key
                            </span>
                            {showTutorial ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        </button>

                        {showTutorial && (
                            <div className="p-4 bg-white border-t border-black text-xs space-y-2.5">
                                <ol className="list-decimal list-inside space-y-1.5 text-black">
                                    <li>
                                        Visit <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="underline font-bold inline-flex items-center gap-0.5">
                                            Google AI Studio <ExternalLink size={10} />
                                        </a>
                                    </li>
                                    <li>Click <strong>"Create API key"</strong>.</li>
                                    <li>Select or create a project.</li>
                                    <li>Copy the generated key (starts with <code className="font-mono bg-neutral-100 px-1 border border-black">AIzaSy...</code>).</li>
                                    <li>Paste it in the field above.</li>
                                </ol>
                                <p className="text-[11px] text-neutral-600 pt-2 border-t border-black">
                                    Your key is stored only in local storage on your machine.
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="px-6 py-4 bg-white border-t-2 border-black flex justify-between items-center gap-3">
                    {apiKey && hasSavedKey ? (
                        <button
                            onClick={handleRemove}
                            className="px-3 py-2 border border-black text-black bg-white hover:bg-black hover:text-white text-xs font-bold transition-none flex items-center gap-1.5"
                        >
                            <Trash2 size={14} /> Remove Key
                        </button>
                    ) : <div />}

                    <div className="flex gap-2">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 border border-black text-black bg-white hover:bg-black hover:text-white text-xs font-bold transition-none"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={!apiKey.trim()}
                            className="px-5 py-2 border-2 border-black bg-black text-white hover:bg-white hover:text-black text-xs font-bold transition-none disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
                        >
                            <Save size={14} /> Save Key
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default ApiKeyModal;
