import { ArrowRight, Trash2 } from 'lucide-react';
import './HomePage.css';
import { GraphlyLogo } from './GraphlyLogo';

const starters = [
    { title: '2D graph', description: 'Equations & data', label: 'Start graphing', icon: 'y = f(x)', action: 'onCreate2D' },
    { title: '3D graph', description: 'Surfaces & space', label: 'Explore in 3D', icon: 'F(x, y, z) = 0', action: 'onCreate3D' },
    { title: 'Import data', description: 'CSV & image scan', label: 'Add your data', icon: 'x, y', action: 'onImport' },
];

export function HomePage({ onCreate2D, onCreate3D, onImport, savedGraphs = [], onOpenGraph, onDeleteGraph }) {
    const actions = { onCreate2D, onCreate3D, onImport };
    return (
        <main className="graphly-home">
            <svg className="graphly-home-background" viewBox="0 0 1200 700" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
                <g stroke="#e4e4e7" strokeWidth="1">
                    {Array.from({ length: 25 }, (_, i) => <path key={`v${i}`} d={`M${i * 50} 0V700`} />)}
                    {Array.from({ length: 15 }, (_, i) => <path key={`h${i}`} d={`M0 ${i * 50}H1200`} />)}
                </g>
                <path className="graphly-home-wave" d="M-100 370C50 120 150 120 300 370S550 620 700 370S950 120 1100 370S1350 620 1500 370" />
                <path className="graphly-home-wave graphly-home-wave-secondary" d="M-100 520Q300 -100 700 520T1500 520" />
            </svg>
            <div className="graphly-home-content">
                <header className="graphly-home-intro">
                    <div className="graphly-home-brand"><GraphlyLogo size={42} /></div>
                    <h1>See the math.</h1>

                </header>
                <section className="graphly-home-starters" aria-label="Create a graph">
                    {starters.map(({ title, description, label, icon, action }) => (
                        <button key={action} type="button" className="graphly-home-starter" onClick={actions[action]}>
                            <span className="graphly-home-icon">{icon}</span>
                            <h2>{title}</h2>
                            <p>{description}</p>
                            <span className="graphly-home-action">{label}<ArrowRight size={16} aria-hidden="true" /></span>
                        </button>
                    ))}
                </section>
                <section className="graphly-home-projects" aria-labelledby="graphly-projects-heading">
                    <div className="graphly-home-section-heading">
                        <h2 id="graphly-projects-heading">Your saved graphs <span>{savedGraphs.length}</span></h2>
                        <p>Stored in this browser</p>
                    </div>
                    {savedGraphs.length ? (
                        <ul className="graphly-home-project-list">
                            {savedGraphs.map(graph => {
                                const title = graph.title || 'Untitled graph';
                                const count = graph.datasets?.length ?? 0;
                                return (
                                    <li key={graph.id} className="graphly-home-project">
                                        <button type="button" className="graphly-home-open" onClick={() => onOpenGraph(graph)}>

                                            <span><strong>{title}</strong><small>{count} {count === 1 ? 'dataset' : 'datasets'}</small></span>
                                        </button>
                                        <button type="button" className="graphly-home-delete" onClick={event => onDeleteGraph(event, graph.id)} aria-label={`Delete ${title}`} title={`Delete ${title}`}>
                                            <Trash2 size={16} aria-hidden="true" />
                                        </button>
                                    </li>
                                );
                            })}
                        </ul>
                    ) : <p className="graphly-home-empty">Saved graphs will appear here.</p>}
                </section>
            </div>
        </main>
    );
}
