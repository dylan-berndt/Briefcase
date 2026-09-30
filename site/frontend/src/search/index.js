import './main.css'
import React, { useState, useRef, useEffect, useCallback } from 'react';

const PAGE_SIZE = 24;

async function postJSON(url, body) {
	const response = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
	let json = {};
	try { json = await response.json(); } catch (e) { /* no body */ }
	return { ok: response.ok, status: response.status, json };
}

// Numbers to show in the pager: first, last, and a window around the current page, with gaps as null
export function pageWindow(page, totalPages, radius = 2) {
	const pages = new Set([1, totalPages]);
	for (let p = page - radius; p <= page + radius; p++) {
		if (p >= 1 && p <= totalPages) pages.add(p);
	}
	const sorted = [...pages].sort((a, b) => a - b);
	const out = [];
	sorted.forEach((p, i) => {
		if (i > 0 && p - sorted[i - 1] > 1) out.push(null);
		out.push(p);
	});
	return out;
}

export function Pagination({ page, totalPages, total, onPage }) {
	if (totalPages <= 1) return null;
	return <nav className="Pagination" aria-label="Pages of results">
		<button onClick={() => onPage(page - 1)} disabled={page <= 1}>Previous</button>
		{pageWindow(page, totalPages).map((p, i) => p === null
			? <span key={"gap" + i} className="PageGap">…</span>
			: <button key={p} onClick={() => onPage(p)} aria-current={p === page ? "page" : undefined}
				className={p === page ? "PageCurrent" : ""}>{p}</button>)}
		<button onClick={() => onPage(page + 1)} disabled={page >= totalPages}>Next</button>
		<span className="PageCount">Page {page} of {totalPages} · {total} fonts</span>
	</nav>;
}

function Stars({ rating, onRate }) {
	const [hover, setHover] = useState(0);
	const shown = hover || rating.mine || 0;
	return <div className="Stars" onMouseLeave={() => setHover(0)}>
		{[1, 2, 3, 4, 5].map(n => <button key={n} type="button"
			className={n <= shown ? "Star StarOn" : "Star"}
			aria-label={`Rate ${n} star${n > 1 ? "s" : ""}`} aria-pressed={rating.mine === n}
			onMouseEnter={() => setHover(n)}
			onClick={() => onRate(rating.mine === n ? 0 : n)}>★</button>)}
		<span className="RatingSummary">
			{rating.count > 0 ? `${rating.average} (${rating.count})` : "unrated"}
		</span>
	</div>;
}

function Result({ result, query, username, onNeedLogin }) {
	const [vote, setVote] = useState(result.vote);
	const [rating, setRating] = useState(result.rating);
	const [message, setMessage] = useState("");
	const [describing, setDescribing] = useState(false);
	const [description, setDescription] = useState("");

	// Runs a feedback request; a 401 means the session ended, so ask the user to log in again
	async function send(url, body, onSuccess) {
		if (!username) {
			onNeedLogin();
			setMessage("Log in to give feedback");
			return;
		}
		setMessage("");
		try {
			const { ok, status, json } = await postJSON(url, body);
			if (ok) onSuccess(json);
			else if (status === 401) { onNeedLogin(); setMessage("Your session ended, please log in again"); }
			else setMessage(json.message || "Something went wrong");
		} catch (error) {
			setMessage("Could not reach the server");
		}
	}

	const castVote = (value) => send('/api/font/approve', { fontKey: result.key, query, vote: vote === value ? 0 : value },
		json => setVote(json.vote));
	const rate = (stars) => send('/api/font/rate', { fontKey: result.key, rating: stars}, json => setRating(json.rating));
	const describe = (e) => {
		e.preventDefault();
		send('/api/font/describe', { fontKey: result.key, description },
			() => { setMessage("Thanks, description saved"); setDescription(""); setDescribing(false); });
	};

	return <article className="ResultWindow">
		<a href={result.url} target="_blank" rel="noopener noreferrer" className="Specimen">
			<img src={result.specimen} alt={`${result.name} specimen`} width="640" height="180" loading="lazy" />
		</a>
		<div className="ResultInfo">
			<div className="ResultTitle">
				<a href={result.url} target="_blank" rel="noopener noreferrer">{result.name}</a>
				<span className="ResultSource">{result.source === "google" ? "Google Fonts" : "DaFont"}{result.creator ? ` · ${result.creator}` : ""}</span>
			</div>
			<div className="ResultActions">
				<div className="Votes" role="group" aria-label="Does this font answer your search?">
					<button type="button" aria-pressed={vote === 1} className={vote === 1 ? "VoteOn" : ""}
						aria-label="This font matches my search" title="Matches my search" onClick={() => castVote(1)}>Matches</button>
					<button type="button" aria-pressed={vote === -1} className={vote === -1 ? "VoteOn" : ""}
						aria-label="This font does not match my search" title="Doesn't match my search" onClick={() => castVote(-1)}>Doesn't match</button>
				</div>
				<Stars rating={rating} onRate={rate} />
				<button type="button" className="DescribeToggle" onClick={() => setDescribing(!describing)}>Describe</button>
			</div>
		</div>
		{!describing ? null :
			<form className="DescriptionField" onSubmit={describe}>
				<p>How would you describe this font?</p>
				<input type="text" aria-label="Font description" maxLength={500} value={description}
					onChange={e => setDescription(e.target.value)} />
			</form>}
		{message ? <p className="ResultMessage" role="status">{message}</p> : null}
	</article>;
}

function queryDescription(tags) {
	return tags.map(t => (t.weight < 0 ? "not " : "") + t.tag).join(", ");
}

export default function SearchPage({ username, onNeedLogin = () => {} }) {
	const initial = new URLSearchParams(window.location.search);
	const [text, setText] = useState(initial.get("q") || "");
	const [query, setQuery] = useState(initial.get("q") || "");
	const [page, setPage] = useState(Math.max(1, parseInt(initial.get("page"), 10) || 1));
	const [data, setData] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState("");
	const topRef = useRef(null);
	const generation = useRef(0);

	useEffect(() => {
		if (!query.trim()) { setData(null); return undefined; }
		const controller = new AbortController();
		setLoading(true);
		setError("");
		const params = new URLSearchParams({ query, page, pageSize: PAGE_SIZE });
		fetch('/api/font/query?' + params, { signal: controller.signal })
			.then(async response => {
				const json = await response.json();
				if (!response.ok) throw new Error(json.message || "Search failed");
				return json;
			})
			.then(json => {
				// Paged past the end, e.g. from an old link: go to the last page that exists
				if (json.results.length === 0 && json.totalPages > 0 && page > json.totalPages) {
					setPage(json.totalPages);
					return;
				}
				generation.current += 1;
				setData({ ...json, generation: generation.current });
				setLoading(false);
			})
			.catch(e => {
				if (e.name === "AbortError") return;
				setError(e.message || String(e));
				setLoading(false);
			});
		return () => controller.abort();
	}, [query, page, username]);

	const navigate = useCallback((nextQuery, nextPage) => {
		const params = new URLSearchParams();
		if (nextQuery) params.set("q", nextQuery);
		if (nextPage > 1) params.set("page", nextPage);
		const search = params.toString();
		window.history.pushState({}, "", window.location.pathname + (search ? "?" + search : ""));
		setQuery(nextQuery);
		setPage(nextPage);
	}, []);

	// Back and forward buttons
	useEffect(() => {
		const onPop = () => {
			const params = new URLSearchParams(window.location.search);
			setText(params.get("q") || "");
			setQuery(params.get("q") || "");
			setPage(Math.max(1, parseInt(params.get("page"), 10) || 1));
		};
		window.addEventListener("popstate", onPop);
		return () => window.removeEventListener("popstate", onPop);
	}, []);

	const goToPage = (p) => {
		navigate(query, p);
		if (topRef.current && topRef.current.scrollIntoView) topRef.current.scrollIntoView({ block: "start" });
	};

	const fontClasses = ["pixelify", "aldrich", "google", "montserrat", "alfa", "montserrat", "google", "aldrich"];
	const [fontNum, setFontNum] = useState(0);

	useEffect(() => {
		const id = setInterval(() => {
			setFontNum(n => (n + 1) % fontClasses.length);
		}, 300);
		return () => clearInterval(id);
	}, [fontClasses.length]);

	const displayFont = fontClasses[fontNum];

	return (
	<div className="Center">
		<div style={{ height: "10vmin", display: "flex", alignItems: "center", justifyContent: "center" }}>
			<p style={{ fontSize: "6vmin", lineHeight: 1.8, textShadow: "black 0 10px 10px", marginTop: "-12vmin" }} className={displayFont}>
				Font Search <br></br>
			</p>
		</div>
		<div style={{ height: "6vmin" }}></div>
		<p style={{ fontSize: "3vmin", marginBottom: "4vh" }}>
			Describe the font you want, for example "elegant script, not too thin"
		</p>

		<form className="SearchForm" onSubmit={e => { e.preventDefault(); if (text.trim()) navigate(text.trim(), 1); }}>
			<input type="text" name="description" aria-label="Describe a font" value={text}
				onChange={e => setText(e.target.value)} maxLength={200} />
			<button type="submit">Search</button>
		</form>

		<div ref={topRef} className="ResultsTop"></div>
		{error ? <p className="SearchMessage" role="alert">{error}</p> : null}

		{data === null && loading ? <p className="SearchMessage" role="status">Searching…</p> : null}
		{data === null ? null : <>
			<p className="SearchMessage" role="status">
				{data.tags.length > 0
					? `Searching for: ${queryDescription(data.tags)}`
					: "No tags recognised in that description."}
				{data.unmatched.length > 0 ? ` Not recognised: ${data.unmatched.join(", ")}.` : ""}
			</p>
			<div className={loading ? "Results ResultsLoading" : "Results"} aria-busy={loading}>
				{data.results.map(result =>
					<Result key={data.generation + "|" + result.key} result={result} query={query}
						username={username} onNeedLogin={onNeedLogin} />)}
			</div>
			<Pagination page={data.page} totalPages={data.totalPages} total={data.total} onPage={goToPage} />
		</>}
	</div>
	)
}
