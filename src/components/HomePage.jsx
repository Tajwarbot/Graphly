import { ArrowRight, Box, ChartNoAxesCombined, FileUp, FolderOpen, Trash2 } from 'lucide-react';
import './HomePage.css';

const starters = [
    { title: '2D graph', description: 'Explore equations, plot points, and fit your data.', label: 'Start graphing', icon: <ChartNoAxesCombined size={23} aria-hidden="true" />, action: 'onCreate2D' },
    { title: '3D graph', description: 'Explore surfaces and equations from every angle.', label: 'Explore in 3D', icon: <Box size={23} aria-hidden="true" />, action: 'onCreate3D' },
    { title: 'Import data', description: 'Bring in a CSV file or scan a table from an image.', label: 'Add your data', icon: <FileUp size={23} aria-hidden="true" />, action: 'onImport' },
];

export function HomePage({ onCreate2D, onCreate3D, onImport, savedGraphs = [], onOpenGraph, onDeleteGraph }) {
    const actions = { onCreate2D, onCreate3D, onImport };
    return (
        <main className="graphly-home">
            <div className="graphly-home-content">
                <header className="graphly-home-intro">
                    <p className="graphly-home-eyebrow">Your mathematical workspace</p>
                    <h1>A little curiosity.<br /><span>A clearer picture.</span></h1>
                    <p>Turn equations and data into something you can explore. Start a graph, discover a surface, or pick up where you left off.</p>
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
                                            <FolderOpen size={20} aria-hidden="true" />
                                            <span><strong>{title}</strong><small>{count} {count === 1 ? 'dataset' : 'datasets'}</small></span>
                                        </button>
                                        <button type="button" className="graphly-home-delete" onClick={event => onDeleteGraph(event, graph.id)} aria-label={`Delete ${title}`} title={`Delete ${title}`}>
                                            <Trash2 size={16} aria-hidden="true" />
                                        </button>
                                    </li>
                                );
                            })}
                        </ul>
                    ) : (
                        <div className="graphly-home-empty"><FolderOpen size={24} aria-hidden="true" /><div><h3>Room for your next idea</h3><p>Graphs you save will appear here, ready to explore again.</p></div></div>
                    )}
                </section>
            </div>
        </main>
    );
}
