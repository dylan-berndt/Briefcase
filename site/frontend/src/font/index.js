import './main.css'
import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const PREFIX = "/font/";

// The font is named in the address: /font/Abril%20Fatface. A name can hold a "/", which arrives as %2F, so the route is a
// splat and the name is decoded here rather than by the router.
export function slugFromPath(pathname) {
	const raw = pathname.slice(PREFIX.length).replace(/\/$/, "");
	try {
		return decodeURIComponent(raw);
	} catch (e) {
		return raw;
	}
}

// the address of a font's page
export const fontPath = (slug) => PREFIX + encodeURIComponent(slug);

// a square with an arrow leaving its top right corner: this opens somewhere else
function ExternalIcon() {
	return <svg className="FontExternal" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false">
		<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14 21 3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="square" />
	</svg>;
}

function TagList({ title, tags, close }) {
	if (!tags.length) return null;
	return <section className="FontTagGroup">
		<h2>{title}</h2>
		<ul className={"FontTags" + (close ? " FontTagsClose" : "")}>
			{tags.map(tag => <li key={tag}><Link to={"/?q=" + encodeURIComponent(tag)}>{tag}</Link></li>)}
		</ul>
	</section>;
}

// One font: its specimen, the tags it was given and the ones close to it, and the fonts most like it. The server renders
// the same page as plain HTML for search engines (site/backend/seo.py); this is what a visitor gets, from the same data.
export default function FontPage() {
	const slug = slugFromPath(useLocation().pathname);
	const [state, setState] = useState({ slug: null, font: null, missing: false });

	useEffect(() => {
		const controller = new AbortController();
		fetch('/api/font/info/' + encodeURIComponent(slug), { signal: controller.signal })
			.then(response => response.ok ? response.json() : Promise.reject(response.status))
			.then(font => setState({ slug, font, missing: false }))
			.catch(error => { if (!error || error.name !== "AbortError") setState({ slug, font: null, missing: true }); });
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
		<div className="FontLayout">
			<article className="FontMain">
				<h1>
					<a href={font.url} target="_blank" rel="noopener noreferrer">{font.name}<ExternalIcon /></a>
				</h1>
				<p className="FontSource">{font.source}{font.creator ? ` · ${font.creator}` : ""}</p>
				<div className="FontSpecimen">
					<img src={font.specimen} alt={`${font.name} specimen`} width="640" height="180" />
				</div>
				{font.tags.length ? <p><Link to={"/?q=" + encodeURIComponent(font.tags.join(" "))}>Search for more fonts like this</Link></p> : null}
			</article>
			<aside className="FontTagColumn">
				<TagList title="Tags" tags={font.tags} />
				<TagList title="Close tags" tags={font.closeTags} close />
			</aside>
		</div>
		{font.related.length ? <section className="FontSimilar">
			<h2>Similar fonts</h2>
			<ul className="FontCards">
				{font.related.map(other => <li key={other.slug}>
					<Link to={other.path}>
						<img src={other.specimen} alt={`${other.name} specimen`} width="640" height="180" loading="lazy" />
						<span>{other.name}</span>
					</Link>
				</li>)}
			</ul>
		</section> : null}
		<section className="FontAbout">
			<h2>About Font Search</h2>
			{font.about.map((paragraph, i) => <p key={i}>{paragraph}</p>)}
		</section>
	</div>;
}
