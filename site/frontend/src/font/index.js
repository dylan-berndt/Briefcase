import './main.css'
import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';

// One font: its specimen, the tags it was given, and the fonts most like it. The server renders the same page as
// plain HTML for search engines (site/backend/seo.py); this is what a visitor gets, from the same data.
export default function FontPage() {
	const { slug } = useParams();
	const [state, setState] = useState({ slug: null, font: null, missing: false });

	useEffect(() => {
		const controller = new AbortController();
		fetch('/api/font/info/' + encodeURIComponent(slug), { signal: controller.signal })
			.then(response => response.ok ? response.json() : Promise.reject(response.status))
			.then(font => setState({ slug, font, missing: false }))
			.catch(error => { if (error !== undefined && error.name !== "AbortError") setState({ slug, font: null, missing: true }); });
		return () => controller.abort();
	}, [slug]);

	const font = state.slug === slug ? state.font : null;

	useEffect(() => {
		if (!font) return;
		const before = document.title;
		document.title = font.title;
		return () => { document.title = before; };
	}, [font]);

	if (state.slug === slug && state.missing) {
		return <div className="FontPage"><p className="FontMessage">There is no font at this address. <Link to="/">Search for one</Link>.</p></div>;
	}
	if (!font) return <div className="FontPage"><p className="FontMessage" role="status">Loading…</p></div>;

	return <div className="FontPage">
		<article className="FontMain">
			<h1>{font.name}</h1>
			<p className="FontSource">A free font from {font.source}{font.creator ? ` by ${font.creator}` : ""}</p>
			<a href={font.url} target="_blank" rel="noopener noreferrer" className="FontSpecimen">
				<img src={font.specimen} alt={`${font.name} specimen`} width="640" height="180" />
			</a>
			{font.tags.length ? <>
				<h2>What it looks like</h2>
				<ul className="FontTags">
					{font.tags.map(tag => <li key={tag}><Link to={"/?q=" + encodeURIComponent(tag)}>{tag}</Link></li>)}
				</ul>
				<p><Link to={"/?q=" + encodeURIComponent(font.tags.slice(0, 3).join(" "))}>Search for more fonts like this</Link></p>
			</> : null}
			<p><a href={font.url} target="_blank" rel="noopener noreferrer">Get {font.name} on {font.source}</a>, where its license and download are.</p>
		</article>
		{font.related.length ? <section>
			<h2>Fonts like {font.name}</h2>
			<ul className="FontCards">
				{font.related.map(other => <li key={other.slug}>
					<Link to={"/font/" + other.slug}>
						<img src={other.specimen} alt={`${other.name} specimen`} width="640" height="180" loading="lazy" />
						<span>{other.name}</span>
					</Link>
				</li>)}
			</ul>
		</section> : null}
		<nav className="FontSiblings" aria-label="Neighbouring fonts">
			{font.previous ? <Link to={"/font/" + font.previous.slug} rel="prev">← {font.previous.name}</Link> : <span />}
			{font.next ? <Link to={"/font/" + font.next.slug} rel="next">{font.next.name} →</Link> : <span />}
		</nav>
		<p className="FontAbout">Font Search ranks free fonts from Google Fonts and DaFont by describing them in plain words. The tags
			come from a model that looks at each font’s letters, so some are wrong. <Link to="/about">How it works</Link>.</p>
	</div>;
}
