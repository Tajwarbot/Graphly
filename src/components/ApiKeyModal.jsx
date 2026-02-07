import React, { useState, useEffect } from 'react';
import { X, Key, Save, ExternalLink, Check, AlertCircle, Eye, EyeOff, Trash2, HelpCircle, ChevronRight, ChevronDown } from 'lucide-react';
import { getGeminiApiKey, setGeminiApiKey, removeGeminiApiKey } from '../lib/security';

export function ApiKeyModal({ isOpen, onClose, onSave }) {
    const [apiKey, setApiKey] = useState('');
    const [showKey, setShowKey] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [showTutorial, setShowTutorial] = useState(false);

    useEffect(() => {
        if (isOpen) {
            const currentKey = getGeminiApiKey();
            // Don't show the full key if it's from env, but user can overwrite local
            // If local key exists, show it (masked)
            const localKey = localStorage.getItem('graphly_api_key');
            if (localKey) {
                setApiKey(localKey);
            } else {
                setApiKey('');
            }
            setError('');
            setSuccess('');
            setShowTutorial(false);
        }
    }, [isOpen]);

    const handleSave = () => {
        if (!apiKey.trim()) {
            setError('Please enter a valid API key');
            return;
        }

        if (apiKey.length < 30) {
            setError('API key appears to be too short. Please check again.');
            return;
        }

        setGeminiApiKey(apiKey.trim());
        setSuccess('API key saved successfully!');
        setError('');

        // Brief delay before closing to show success message
        setTimeout(() => {
            onSave();
            onClose();
        }, 1000);
    };

    const handleRemove = () => {
        if (confirm('Are you sure you want to remove your API key? You will need to enter it again to use AI features.')) {
            removeGeminiApiKey();
            setApiKey('');
            setSuccess('API key removed.');
            setTimeout(() => {
                setSuccess('');
            }, 2000);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div
                className="bg-white rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden animate-in zoom-in-95 duration-200 border border-indigo-100"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="px-6 py-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                    <div className="flex items-center gap-2 text-indigo-900">
                        <div className="bg-indigo-100 p-2 rounded-lg">
                            <Key size={20} className="text-indigo-600" />
                        </div>
                        <h2 className="text-lg font-bold">API Key Required</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-200 rounded-full text-slate-400 hover:text-slate-600 transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-6">
                    <div className="bg-indigo-50/50 border border-indigo-100 rounded-xl p-4 text-sm text-indigo-900 leading-relaxed">
                        <p>
                            To process images and extract data, Graphly uses Google's Gemini AI.
                            You need to provide your own free API key to continue.
                        </p>
                    </div>

                    {/* Input Field */}
                    <div className="space-y-2">
                        <label className="text-sm font-bold text-slate-700 block ml-1">
                            Enter your Gemini API Key
                        </label>
                        <div className="relative">
                            <input
                                type={showKey ? "text" : "password"}
                                value={apiKey}
                                onChange={(e) => {
                                    setApiKey(e.target.value);
                                    setError('');
                                }}
                                placeholder="AIzaSy..."
                                className={`
                                    w-full pl-4 pr-12 py-3 bg-white border rounded-xl outline-none transition-all
                                    focus:ring-2 focus:ring-indigo-100 
                                    ${error ? 'border-red-300 focus:border-red-400' : 'border-slate-200 focus:border-indigo-400'}
                                    text-slate-700 font-mono text-sm
                                `}
                            />
                            <button
                                type="button"
                                onClick={() => setShowKey(!showKey)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                            >
                                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                            </button>
                        </div>

                        {/* Error / Success Messages */}
                        {error && (
                            <div className="flex items-center gap-2 text-red-500 text-sm mt-2 ml-1 animate-in slide-in-from-left-2">
                                <AlertCircle size={14} />
                                <span>{error}</span>
                            </div>
                        )}
                        {success && (
                            <div className="flex items-center gap-2 text-green-600 text-sm mt-2 ml-1 animate-in slide-in-from-left-2">
                                <Check size={14} />
                                <span>{success}</span>
                            </div>
                        )}
                    </div>

                    {/* Tutorial Accordion */}
                    <div className="border border-slate-200 rounded-xl overflow-hidden">
                        <button
                            onClick={() => setShowTutorial(!showTutorial)}
                            className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-slate-100 transition-colors text-left group"
                        >
                            <span className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                                <HelpCircle size={16} className="text-indigo-500" />
                                How to get a free API Key
                            </span>
                            {showTutorial ?
                                <ChevronDown size={16} className="text-slate-400" /> :
                                <ChevronRight size={16} className="text-slate-400 group-hover:translate-x-0.5 transition-transform" />
                            }
                        </button>

                        {showTutorial && (
                            <div className="p-4 bg-white border-t border-slate-200 text-sm space-y-3 animate-in slide-in-from-top-2">
                                <ol className="list-decimal list-inside space-y-2 text-slate-600 ml-1">
                                    <li>
                                        Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline inline-flex items-center gap-0.5 font-medium">
                                            Google AI Studio <ExternalLink size={10} />
                                        </a>
                                    </li>
                                    <li>Click on <strong>"Create API key"</strong>.</li>
                                    <li>Select a project (or create a new one).</li>
                                    <li>Copy the generated key (starts with <code>AIzaSy...</code>).</li>
                                    <li>Paste it in the field above.</li>
                                </ol>
                                <p className="text-xs text-slate-400 pt-2 border-t border-slate-100">
                                    Note: Graphly stores your key locally in your browser. It is never sent to our servers, only directly to Google's API.
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex justify-between items-center gap-3">
                    {apiKey && localStorage.getItem('graphly_api_key') ? (
                        <button
                            onClick={handleRemove}
                            className="px-4 py-2.5 text-red-500 hover:text-red-700 hover:bg-red-50 rounded-xl text-sm font-medium transition-colors flex items-center gap-2"
                        >
                            <Trash2 size={16} /> Remove Key
                        </button>
                    ) : <div></div>}

                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2.5 text-slate-600 hover:text-slate-800 hover:bg-slate-200 rounded-xl text-sm font-medium transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={!apiKey.trim()}
                            className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-sm font-medium shadow-sm hover:shadow-md transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                        >
                            <Save size={16} /> Save Key
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default ApiKeyModal;
