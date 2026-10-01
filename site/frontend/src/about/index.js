import './main.css'
import React, { useState, useEffect, useMemo } from 'react';
import { Marked } from 'marked';
import aboutUrl from './about.md';

// The page is about.md, rendered. Edit that file; nothing else needs to change. The table of contents at the top is
// made from its ## and ### headings (the # heading is the page title and is left out), and is left out entirely
// when the file has fewer than two of them.
const TOC_DEPTHS = [2, 3];

const escapeAttr = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export function slugify(text, seen) {
	const base = text.toLowerCase().replace(/<[^>]*>/g, "").replace(/&[a-z#0-9]+;/g, "")
		.replace(/[^\p{L}\p{N}\s-]/gu, "").trim().replace(/\s+/g, "-") || "section";
	let slug = base;
	for (let n = 2; seen.has(slug); n++) slug = `${base}-${n}`;
	seen.add(slug);
	return slug;
}

// markdown text -> {titleHtml, html, toc: [{id, text, depth}]}. A first-line # heading is split off as the title so the
// table of contents can sit between it and the text.
export function renderMarkdown(text) {
	const marked = new Marked({
		renderer: {
			heading(token) {
				const inner = this.parser.parseInline(token.tokens);
				return `<h${token.depth} id="${token.slug}">${inner}</h${token.depth}>\n`;
			},
			link(token) {
				const inner = this.parser.parseInline(token.tokens);
				const title = token.title ? ` title="${escapeAttr(token.title)}"` : "";
				const href = token.href.replace(/"/g, "&quot;").replace(/</g, "%3C").replace(/>/g, "%3E");
				const external = /^https?:\/\//i.test(token.href) ? ' target="_blank" rel="noopener noreferrer"' : "";
				return `<a href="${href}"${title}${external}>${inner}</a>`;
			},
		},
	});
	const tokens = marked.lexer(text);
	const seen = new Set();
	const toc = [];
	for (const token of tokens) {
		if (token.type !== "heading") continue;
		token.slug = slugify(token.text, seen);
		if (TOC_DEPTHS.includes(token.depth)) toc.push({ id: token.slug, text: token.text.replace(/[*_`]/g, ""), depth: token.depth });
	}
	const hasTitle = tokens.length > 0 && tokens[0].type === "heading" && tokens[0].depth === 1;
	return {
		titleHtml: hasTitle ? marked.parser([tokens[0]]) : "",
		html: marked.parser(hasTitle ? tokens.slice(1) : tokens),
		toc,
	};
}

export default function AboutPage() {
	const [source, setSource] = useState(null);
	const [error, setError] = useState("");

	useEffect(() => {
		const controller = new AbortController();
		fetch(aboutUrl, { signal: controller.signal })
			.then(response => {
				if (!response.ok) throw new Error("Could not load the page");
				return response.text();
			})
			.then(setSource)
			.catch(e => { if (e.name !== "AbortError") setError(e.message || String(e)); });
		return () => controller.abort();
	}, []);

	const page = useMemo(() => source === null ? null : renderMarkdown(source), [source]);

	const goTo = (event, id) => {
		const target = document.getElementById(id);
		if (!target) return;
		event.preventDefault();
		target.scrollIntoView({ behavior: "smooth", block: "start" });
	};

	return <div className="About">
		{error ? <p className="AboutMessage" role="alert">{error}</p> : null}
		{page === null && !error ? <p className="AboutMessage" role="status">Loading…</p> : null}
		{page === null ? null : <article className="AboutBody">
			{page.titleHtml ? <div className="AboutText AboutTitle" dangerouslySetInnerHTML={{ __html: page.titleHtml }} /> : null}
			{page.toc.length >= 2 ? <nav className="AboutContents" aria-label="Table of contents">
				<p className="AboutContentsTitle">Contents</p>
				<ol>
					{page.toc.map(entry => <li key={entry.id} className={"AboutContents-" + entry.depth}>
						<a href={"#" + entry.id} onClick={e => goTo(e, entry.id)}>{entry.text}</a>
					</li>)}
				</ol>
			</nav> : null}
			<div className="AboutText" dangerouslySetInnerHTML={{ __html: page.html }} />
		</article>}
	</div>;
}
