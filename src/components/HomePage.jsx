import { MathBackground } from './MathBackground';
import { Trash2 } from 'lucide-react';
import './HomePage.css';
import { GraphlyLogo } from './GraphlyLogo';

const starters = [
    { title: '2D Plotter', description: 'Equations & data', label: 'Start graphing', icon: 'y = f(x)', action: 'onCreate2D' },
    { title: '3D Surface', description: 'Surfaces & space', label: 'Explore in 3D', icon: 'F(x, y, z) = 0', action: 'onCreate3D' },
    { title: 'Scan & CSV', description: 'CSV & image scan', label: 'Add your data', icon: 'x, y', action: 'onImport' },
];

export function HomePage({ onCreate2D, onCreate3D, onImport, savedGraphs = [], onOpenGraph, onDeleteGraph }) {
    const actions = { onCreate2D, onCreate3D, onImport };
    return (
        <main className="graphly-home">
            <MathBackground />
            <div className="graphly-home-content">
                <header className="graphly-home-intro">
                    <div className="graphly-home-brand"><GraphlyLogo size={42} /></div>
                    <h1>Precision Coordinate Plotter</h1>
                    <p>Mathematical curves, experimental data, and interactive 3D surfaces.</p>

                </header>
                <section className="graphly-home-starters" aria-label="Create a graph">
                    {starters.map(({ title, description, icon, action }) => (
                        <button key={action} type="button" className="graphly-home-starter" onClick={actions[action]}>
                            <span className="graphly-home-icon">{icon}</span>
                            <h2>{title}</h2>
                            <p>{description}</p>
                        </button>
                    ))}
                </section>
                {savedGraphs.length > 0 && <section className="graphly-home-projects" aria-labelledby="graphly-projects-heading">
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
                </section>}
            </div>
        </main>
    );
}
