import './main.css'
import React, { useState, useEffect, useMemo } from 'react';
import { Marked } from 'marked';
import aboutUrl from './about.md';

// The page is about.md, rendered. Edit that file; nothing else needs to change. The table of contents at the top is
// made from the headings whose level is in TOC_DEPTHS (a first-line # heading is the page title and is never listed),
// indented by how much deeper each is than the shallowest one listed, and is left out entirely when fewer than two
// headings qualify.
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
export function renderMarkdown(text, depths = TOC_DEPTHS) {
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
	const hasTitle = tokens.length > 0 && tokens[0].type === "heading" && tokens[0].depth === 1;
	const seen = new Set();
	const toc = [];
	for (const [i, token] of tokens.entries()) {
		if (token.type !== "heading") continue;
		token.slug = slugify(token.text, seen);
		if (depths.includes(token.depth) && !(hasTitle && i === 0)) toc.push({ id: token.slug, text: token.text.replace(/[*_`]/g, ""), depth: token.depth });
	}
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

	const top = page === null || page.toc.length === 0 ? 0 : Math.min(...page.toc.map(entry => entry.depth));

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
					{page.toc.map(entry => <li key={entry.id} className={entry.depth > top ? "AboutContents-nested" : undefined}
						style={{ paddingLeft: (entry.depth - top) * 20 }}>
						<a href={"#" + entry.id} onClick={e => goTo(e, entry.id)}>{entry.text}</a>
					</li>)}
				</ol>
			</nav> : null}
			<div className="AboutText" dangerouslySetInnerHTML={{ __html: page.html }} />
		</article>}
	</div>;
}
