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

function Thumb({ down = false }) {
	return <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"
		style={down ? { transform: "rotate(180deg)" } : undefined}>
		<path fill="currentColor" d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z" />
	</svg>;
}

function Result({ result, query, username, onNeedLogin, allowDescriptions }) {
	const [vote, setVote] = useState(result.vote);
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
			<div className="ResultFeedback">
				<div className="Votes" role="group" aria-label="Does this font answer your query?">
					<button type="button" aria-pressed={vote === 1} className={vote === 1 ? "VoteOn" : ""}
						aria-label="This font matched my query" title="This font matched my query" onClick={() => castVote(1)}>
						<Thumb />
					</button>
					<button type="button" aria-pressed={vote === -1} className={vote === -1 ? "VoteOn" : ""}
						aria-label="This font did not match my query" title="This font did not match my query" onClick={() => castVote(-1)}>
						<Thumb down />
					</button>
				</div>
				{allowDescriptions
					? <button type="button" className="DescribeToggle" onClick={() => setDescribing(!describing)}>Describe</button>
					: null}
			</div>
		</div>
		{allowDescriptions && describing ?
			<form className="DescriptionField" onSubmit={describe}>
				<p>How would you describe this font?</p>
				<input type="text" aria-label="Font description" maxLength={500} value={description}
					onChange={e => setDescription(e.target.value)} />
			</form> : null}
		{message ? <p className="ResultMessage" role="status">{message}</p> : null}
	</article>;
}

// The tags a search uses, on one line. Each is a word with a box that is ticked (included), crossed (excluded) or
// empty (off); clicking steps through tick -> empty -> cross -> empty -> tick. Suggested synonyms are in the same list,
// unticked. The server only lists the tags for a query; the boxes are this page's state.
const STATE_NAME = { on: "included", off: "off", neg: "excluded" };

function StateIcon({ state }) {
	if (state === "on") return <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
		<path d="M2 8.5 6 12.5 14 3.5" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="square" /></svg>;
	if (state === "neg") return <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
		<path d="M3 3 13 13M13 3 3 13" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="square" /></svg>;
	return null;
}

export function TagLine({ tags, unmatched, nextAfterOff, onToggle }) {
	// an empty box becomes a tick or a cross depending on where it came from
	const nextState = (tag) => tag.state === "off" ? (nextAfterOff.current[tag.tag] || "on") : "off";
	return <div className="TagLine">
		{tags.length === 0
			? <span className="TagLabel" role="status">No tags recognised in that description.</span> : null}
		<ul className="Tags" aria-label="Tags in your search">
			{tags.map(tag => {
				const next = nextState(tag);
				return <li key={tag.tag} className="Tag">
					<span className="TagWord">{tag.tag}</span>
					<button type="button" className={"TagBox TagBox-" + tag.state} onClick={() => onToggle(tag, next)}
						aria-label={`${tag.tag}: ${STATE_NAME[tag.state]}`}
						title={`${STATE_NAME[tag.state][0].toUpperCase() + STATE_NAME[tag.state].slice(1)}. Click to ${next === "on" ? "include" : next === "neg" ? "exclude" : "turn off"}.`}>
						<StateIcon state={tag.state} />
					</button>
				</li>;
			})}
		</ul>
		{unmatched.length > 0 ? <span className="TagLabel">Not recognised: {unmatched.join(", ")}.</span> : null}
	</div>;
}

// allowDescriptions shows the per-font description box; hidden for now, the endpoint is still there
export default function SearchPage({ username, onNeedLogin = () => {}, allowDescriptions = false }) {
	const initial = new URLSearchParams(window.location.search);
	const [text, setText] = useState(initial.get("q") || "");
	const [query, setQuery] = useState(initial.get("q") || "");
	const [page, setPage] = useState(Math.max(1, parseInt(initial.get("page"), 10) || 1));
	const [searchId, setSearchId] = useState(0);   // bumps on every submit, so searching the same text starts over
	const [tagSet, setTagSet] = useState(null);    // {tags: [{tag, weight, state}], unmatched}; null while loading
	const [data, setData] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState("");
	const topRef = useRef(null);
	const generation = useRef(0);
	const nextAfterOff = useRef({});

	// 1. The tags a query means. Nothing else about the query is asked of the server again.
	useEffect(() => {
		setTagSet(null);
		if (!query.trim()) { setData(null); return undefined; }
		const controller = new AbortController();
		setLoading(true);
		setError("");
		fetch('/api/font/tags?' + new URLSearchParams({ query }), { signal: controller.signal })
			.then(async response => {
				const json = await response.json();
				if (!response.ok) throw new Error(json.message || "Search failed");
				return json;
			})
			.then(json => setTagSet({
				tags: [...json.tags.map(t => ({ tag: t.tag, weight: Math.abs(t.weight), state: t.weight < 0 ? "neg" : "on" })),
					...json.suggested.map(s => ({ tag: s.tag, weight: 1, state: "off" }))],
				unmatched: json.unmatched,
			}))
			.catch(e => {
				if (e.name === "AbortError") return;
				setError(e.message || String(e));
				setLoading(false);
			});
		return () => controller.abort();
	}, [query, searchId]);

	// 2. The fonts for the tags that are ticked or crossed. query stays only as the label votes are filed under.
	const tagParam = tagSet === null ? null : tagSet.tags.filter(t => t.state !== "off")
		.map(t => `${t.state === "neg" ? "-" : ""}${t.tag}:${t.weight}`).join(",");
	useEffect(() => {
		if (tagParam === null) return undefined;
		const controller = new AbortController();
		setLoading(true);
		setError("");
		const params = new URLSearchParams({ query, tags: tagParam, page, pageSize: PAGE_SIZE });
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
	}, [query, page, tagParam, username]);

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

	// tick -> empty -> cross -> empty -> tick: remember which way an empty box goes next
	const toggleTag = (tag, next) => {
		if (tag.state === "on") nextAfterOff.current[tag.tag] = "neg";
		if (tag.state === "neg") nextAfterOff.current[tag.tag] = "on";
		setTagSet(set => ({ ...set, tags: set.tags.map(t => t.tag === tag.tag ? { ...t, state: next } : t) }));
		if (page !== 1) navigate(query, 1);   // different tags, different results: back to the first page
	};

	const goToPage = (p) => {
		navigate(query, p);
		if (topRef.current && topRef.current.scrollIntoView) topRef.current.scrollIntoView({ block: "start" });
	};

	const fontClasses = ["pixelify", "aldrich", "google", "montserrat", "alfa", "montserrat", "google", "aldrich"];
	const [fontNum, setFontNum] = useState(0);

	useEffect(() => {
		const id = setInterval(() => {
			setFontNum(n => (n + 1) % fontClasses.length);
		}, 1000);
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
		<p style={{ fontSize: "2vmin", marginBottom: "4vh" }}>
			Please enter a description to search for a font
		</p>

		<form className="SearchForm" onSubmit={e => {
			e.preventDefault();
			if (!text.trim()) return;
			nextAfterOff.current = {};
			setSearchId(n => n + 1);
			navigate(text.trim(), 1);
		}}>
			<input type="text" name="description" aria-label="Describe a font" value={text}
				onChange={e => setText(e.target.value)} maxLength={200} />
		</form>

		<div ref={topRef} className="ResultsTop"></div>
		{error ? <p className="SearchMessage" role="alert">{error}</p> : null}

		{tagSet === null && data === null && loading ? <p className="SearchMessage" role="status">Searching…</p> : null}
		{tagSet === null ? null
			: <TagLine tags={tagSet.tags} unmatched={tagSet.unmatched} nextAfterOff={nextAfterOff} onToggle={toggleTag} />}
		{data === null || tagSet === null ? null : <>
			{data.total === 0 && tagSet.tags.length > 0
				? <p className="SearchMessage">Tick a tag to see fonts.</p> : null}
			<div className={loading ? "Results ResultsLoading" : "Results"} aria-busy={loading}>
				{data.results.map(result =>
					<Result key={data.generation + "|" + result.key} result={result} query={query}
						username={username} onNeedLogin={onNeedLogin} allowDescriptions={allowDescriptions} />)}
			</div>
			<Pagination page={data.page} totalPages={data.totalPages} total={data.total} onPage={goToPage} />
		</>}
	</div>
	)
}
