import { symbolicMath } from '../lib/symbolicMath.js';
import React, { useState, useRef, useEffect } from 'react';
import { getGeminiApiKey, isApiKeyConfigured } from '../lib/security';
import { parseIntentLocally } from '../lib/chatIntent.js';
import { AI_TOOLS_DECLARATION } from '../lib/chatTools.js';
import { createGraphChat } from '../lib/aiProvider.js';
import { runChatTools } from '../lib/chatToolLoop.js';
import {
    ATTACHMENT_ACCEPT,
    readAttachment,
    attachmentParts,
    parsePointTable
} from '../lib/chatAttachments.js';
import './AIChatbox.css';

export function AIChatbox(props) {
    const [isOpen, setIsOpen] = useState(false);
    const [hasKey, setHasKey] = useState(false);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [attachments, setAttachments] = useState([]);
    const [error, setError] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [reading, setReading] = useState(false);
    const transcript = useRef(null),
        files = useRef(null),
        latest = useRef(props);
    latest.current = props;
    useEffect(() => {
        const sync = () => setHasKey(isApiKeyConfigured());
        sync();
        window.addEventListener('focus', sync);
        window.addEventListener('graphly-key-change', sync);
        return () => {
            window.removeEventListener('focus', sync);
            window.removeEventListener('graphly-key-change', sync);
        };
    }, [isOpen]);
    useEffect(() => {
        if (transcript.current)
            transcript.current.scrollTop = transcript.current.scrollHeight;
    }, [messages, isProcessing]);

    const executeTool = (name, args) => {
        const p = latest.current;
        try {
            let data;
            switch (name) {
                case 'symbolicMath':
                    data = symbolicMath(args);
                    break;
                case 'add2DScene':
                    data = p.onAdd2DScene(args.layers, args.bounds ? {bounds:args.bounds} : {});
                    break;
                case 'addScene':
                    data = p.onAddScene(args.surfaces, {
                        ...(args.bounds ? { bounds: args.bounds } : {}),
                        ...(args.camera ? { camera: args.camera } : {})
                    });
                    break;
                case 'set3DView':
                    data = p.onSet3DView(args);
                    break;
                case 'setLayerStyle': {
                    const { layerId, ...style } = args;
                    data = p.onSetLayerStyle(layerId, style);
                    break;
                }
                case 'removeLayer':
                    data = p.onRemoveLayer(args.layerId);
                    break;
                // The App handlers validate against the current parameter environment.
                // Compiling here without that environment rejects valid named parameters.
                case 'setParameter': {
                    const { layerId, value, ...settings } = args;
                    data = p.onSetParameter(layerId, value, settings);
                    break;
                }
                case 'updateExpression':
                    data = p.onUpdateExpression(args.layerId, args.expression);
                    break;
                case 'switchTo3D':
                    data = p.onSwitchTo3D(args.expression);
                    break;
                case 'plotFunction':
                    data = p.onPlotFunction(args.expression);
                    break;
                case 'plotImplicitEquation':
                    data = p.onPlotImplicit(args.expression);
                    break;
                case 'loadDataTable':
                    if (
                        !Array.isArray(args.rows) ||
                        !args.rows.length ||
                        args.rows.length > 10000 ||
                        args.rows.some(
                            (r) =>
                                !Number.isFinite(r.x) || !Number.isFinite(r.y)
                        )
                    )
                        throw new Error(
                            'Provide 1–10,000 finite numeric points.'
                        );
                    data = p.onLoadDataTable(args.name || 'Data', args.rows);
                    break;
                case 'setViewportBounds':
                    data = p.onSetViewportBounds(args);
                    break;
                default:
                    throw new Error(`Unknown tool: ${name}`);
            }
            return {
                success: true,
                message:
                    name === 'switchTo3D'
                        ? `Added 3D surface: ${args.expression}`
                        : name === 'add2DScene'
                          ? `Added ${args.layers.length} 2D layers.`
                        : name === 'addScene'
                          ? `Added ${args.surfaces.length} surfaces.`
                          : `Applied ${name}${args.expression ? `: ${args.expression}` : ''}.`,
                ...data
            };
        } catch (err) {
            return { success: false, message: err.message };
        }
    };
    const addFiles = async (event) => {
        const selected = Array.from(event.target.files || []);
        event.target.value = '';
        setError('');
        if (attachments.length + selected.length > 3) {
            setError('Attach up to three files per message.');
            return;
        }
        setReading(true);
        try {
            const loaded = await Promise.all(selected.map(readAttachment));
            setAttachments((previous) => [...previous, ...loaded]);
        } catch (err) {
            setError(err.message);
        } finally {
            setReading(false);
        }
    };
    const handleSend = async (event) => {
        event?.preventDefault();
        const prompt = input.trim();
        if (!prompt || isProcessing || reading) return;
        const pending = [...attachments];
        const apiKey = getGeminiApiKey();
        setError('');
        setInput('');
        setAttachments([]);
        setIsProcessing(true);
        setMessages((previous) => {
            const retained = previous.map(message => ({...message}));
            let kept = pending.length ? 1 : 0;
            for (let i = retained.length - 1; i >= 0; i--) {
                if (retained[i].attachments?.length && ++kept > 2) delete retained[i].attachments;
            }
            return [...retained, {role:'user', text:prompt, files:pending.map(f=>f.name), attachments:pending}];
        });
        try {
            let result;
            if (apiKey) {
                let attachmentTurns = pending.length ? 1 : 0;
                const historyMessages = messages.slice(-20).map(m=>({...m}));
                for (let i=historyMessages.length-1;i>=0;i--) {
                    if (historyMessages[i].attachments?.length && ++attachmentTurns > 2) delete historyMessages[i].attachments;
                }
                const history = historyMessages
                    .map((m) => ({
                        role: m.role === 'assistant' ? 'model' : 'user',
                        parts: [{ text: m.text }, ...attachmentParts(m.attachments || [])]
                    }));
                while (history.length && history[0].role !== 'user')
                    history.shift();
                const chat = createGraphChat({
                    apiKey,
                    history,
                    tools: AI_TOOLS_DECLARATION,
                    context: latest.current.graphContext
                });
                result = await runChatTools(
                    chat,
                    [{ text: prompt }, ...attachmentParts(pending)],
                    executeTool
                );
            } else if (pending.length) {
                if (!/\b(plot|import|graph|load)\b/i.test(prompt))
                    throw new Error(
                        'Connect Gemini to interpret attachments, or ask to plot a CSV/TSV table.'
                    );
                if (pending.some((f) => !['csv', 'tsv'].includes(f.extension)))
                    throw new Error(
                        'Connect Gemini to interpret images and text files. CSV and TSV plotting works without a key.'
                    );
                const tables = pending.map((f) => ({
                    name: f.name,
                    rows: parsePointTable(
                        f.text,
                        f.extension === 'tsv' ? '\t' : ','
                    )
                }));
                const actions = tables.map((args) => {
                    const r = executeTool('loadDataTable', args);
                    return {
                        name: 'loadDataTable',
                        result: r.message,
                        success: r.success
                    };
                });
                result = { text: 'Table import results are below.', actions };
            } else
                result = parseIntentLocally(
                    prompt,
                    executeTool,
                    latest.current.graphContext
                );
            setMessages((previous) => [
                ...previous,
                { role: 'assistant', ...result }
            ]);
        } catch (err) {
            // Do not fall back and replay mutations after a model/network failure.
            const message = /429|quota/i.test(err.message)
                ? 'Gemini usage limit reached. Check your AI Studio quota, then retry.'
                : /API_KEY|not valid|403|401/i.test(err.message)
                  ? 'Gemini rejected the key. Open Connect AI to check it.'
                  : err.message;
            setMessages((previous) => [
                ...previous,
                { role: 'assistant', text: message, actions: [] }
            ]);
            setAttachments(pending);
            setInput(prompt);
        } finally {
            setIsProcessing(false);
        }
    };
    return (
        <div className="graphly-chat-anchor">
            {!isOpen ? (
                <button
                    className="graphly-chat-launch"
                    onClick={() => setIsOpen(true)}
                >
                    Ask Graphly · {isProcessing ? 'Working' : hasKey ? 'AI ready' : 'Offline'}
                </button>
            ) : (
                <section
                    className="graphly-chat"
                    aria-label="Graphly assistant"
                >
                    <header>
                        <div>
                            <strong>Graphly</strong>
                            <span role="status" aria-live="polite">
                                {isProcessing ? 'Working · composing your graph' : hasKey ? 'AI ready · Gemini key configured' : 'Offline · local plotting'}
                            </span>
                        </div>
                        <button
                            aria-label="Close assistant"
                            onClick={() => setIsOpen(false)}
                        >
                            ×
                        </button>
                    </header>
                    <div className="graphly-chat-connection">
                        <button onClick={props.onOpenApiKeyModal}>
                            {hasKey ? 'AI settings' : 'Connect AI'}
                        </button>
                        <span>
                            {hasKey
                                ? 'Images and files are sent with your message.'
                                : 'Use your Google AI Studio key.'}
                        </span>
                    </div>
                    <div
                        className="graphly-chat-transcript"
                        ref={transcript}
                        aria-live="polite"
                    >
                        {!messages.length && (
                            <div className="graphly-chat-intro">
                                <p>What would you like to make?</p>
                                <p>
                                    Describe a shape, combine equations, or
                                    attach data.
                                </p>
                                {[
                                    'Plot two intersecting planes',
                                    'Build a planet with rings',
                                    'Plot a torus'
                                ].map((prompt) => (
                                    <button
                                        key={prompt}
                                        onClick={() => setInput(prompt)}
                                    >
                                        {prompt}
                                    </button>
                                ))}
                            </div>
                        )}
                        {messages.map((m, i) => (
                            <article
                                key={i}
                                className={
                                    m.role === 'user'
                                        ? 'chat-user'
                                        : 'chat-assistant'
                                }
                            >
                                <small>
                                    {m.role === 'user' ? 'You' : 'Graphly'}
                                </small>
                                <p>{m.text}</p>
                                {m.files?.map((name) => (
                                    <span className="chat-file" key={name}>
                                        {name}
                                    </span>
                                ))}
                                {m.actions?.length > 0 && (
                                    <details>
                                        <summary>
                                            {
                                                m.actions.filter(
                                                    (a) => a.success
                                                ).length
                                            }{' '}
                                            updates ·{' '}
                                            {
                                                m.actions.filter(
                                                    (a) => !a.success
                                                ).length
                                            }{' '}
                                            errors
                                        </summary>
                                        {m.actions.map((a, j) => (
                                            <p key={j}>
                                                {a.success
                                                    ? 'Applied: '
                                                    : 'Failed: '}
                                                {a.result}
                                            </p>
                                        ))}
                                    </details>
                                )}
                            </article>
                        ))}
                        {isProcessing && (
                            <p role="status">Working through your request…</p>
                        )}
                    </div>
                    <form onSubmit={handleSend}>
                        {attachments.length > 0 && (
                            <div className="chat-attachments">
                                {attachments.map((file, i) => (
                                    <span key={i}>
                                        {file.name}
                                        <button
                                            type="button"
                                            aria-label={`Remove ${file.name}`}
                                            onClick={() =>
                                                setAttachments((a) =>
                                                    a.filter((_, j) => i !== j)
                                                )
                                            }
                                        >
                                            ×
                                        </button>
                                    </span>
                                ))}
                            </div>
                        )}
                        {error && <p role="alert">{error}</p>}
                        <textarea
                            aria-label="Message Graphly"
                            placeholder="Describe a graph or design…"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (
                                    e.key === 'Enter' &&
                                    !e.shiftKey &&
                                    !e.nativeEvent.isComposing
                                ) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            rows={2}
                        />
                        <div className="chat-compose-actions">
                            <input
                                ref={files}
                                hidden
                                type="file"
                                multiple
                                accept={ATTACHMENT_ACCEPT}
                                onChange={addFiles}
                            />
                            <button
                                type="button"
                                disabled={reading || isProcessing}
                                onClick={() => files.current.click()}
                            >
                                {reading ? 'Reading…' : 'Attach files'}
                            </button>
                            <button
                                type="submit"
                                disabled={
                                    !input.trim() || isProcessing || reading
                                }
                            >
                                Send
                            </button>
                        </div>
                        <small>
                            PNG, JPG, WebP, CSV, TSV, TXT, Markdown, JSON
                        </small>
                    </form>
                </section>
            )}
        </div>
    );
}
