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
	return <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true" focusable="false"
		style={down ? { transform: "rotate(180deg)" } : undefined}>
		<path fill="currentColor" d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z" />
	</svg>;
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

function Result({ result, query, username, onNeedLogin, allowDescriptions }) {
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
			<div className="ResultFeedback">
				<Stars rating={rating} onRate={rate} />
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

const listParam = (params, name) => (params.get(name) || "").split(",").map(x => x.trim()).filter(Boolean);

// What the engine understood, on one line: each tag (or guessed word, or suggested synonym) with a box that is
// ticked (included), crossed (excluded) or empty (off). Clicking steps through tick -> empty -> cross -> empty -> tick.
const STATE_NAME = { on: "included", off: "off", neg: "excluded" };

function StateIcon({ state }) {
	if (state === "on") return <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
		<path d="M2 8.5 6 12.5 14 3.5" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="square" /></svg>;
	if (state === "neg") return <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
		<path d="M3 3 13 13M13 3 3 13" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="square" /></svg>;
	return null;
}

export function TagLine({ data, ignored, nextAfterOff, onChange }) {
	const inferred = data.inferred || [];
	const suggested = data.suggested || [];
	const seen = new Set();
	const chips = [];
	const push = (chip) => { if (!seen.has(chip.name)) { seen.add(chip.name); chips.push(chip); } };
	data.tags.forEach(t => push({ name: t.tag, state: t.weight < 0 ? "neg" : "on" }));
	inferred.forEach(i => push({ name: i.word, state: i.weight < 0 ? "neg" : "on",
		hint: i.tags.join(", "), title: `"${i.word}" is not a tag. Guessed from fonts described that way.` }));
	// turned off: the server no longer reports these, so they are kept from the URL to stay visible
	ignored.forEach(name => push({ name, state: "off" }));
	const offered = suggested.map(({ word, tags }) => ({
		word, chips: tags.filter(t => !seen.has(t.tag)).map(t => ({ name: t.tag, state: "off",
			title: `Similar to “${t.via}” (${t.similarity})` })),
	})).filter(group => group.chips.length > 0);

	const suggestedWords = new Set(suggested.map(s => s.word));
	const notRecognised = data.unmatched.filter(word => !suggestedWords.has(word) && !seen.has(word));

	const nextState = (chip) => chip.state === "off" ? (nextAfterOff.current[chip.name] || "on") : "off";

	const renderChip = (chip, extra = "") => {
		const next = nextState(chip);
		return <li key={chip.name} className={"Tag " + extra} title={chip.title}>
			<span className="TagWord">{chip.name}</span>
			{chip.hint ? <span className="TagHint">→ {chip.hint}</span> : null}
			<button type="button" className={"TagBox TagBox-" + chip.state} onClick={() => onChange(chip, next)}
				aria-label={`${chip.name}: ${STATE_NAME[chip.state]}`}
				title={`${STATE_NAME[chip.state][0].toUpperCase() + STATE_NAME[chip.state].slice(1)}. Click to ${next === "on" ? "include" : next === "neg" ? "exclude" : "turn off"}.`}>
				<StateIcon state={chip.state} />
			</button>
		</li>;
	};

	return <div className="TagLine">
		{chips.length === 0 ? <span className="TagLabel" role="status">No tags recognised in that description.</span>
			: <span className="TagLabel" role="status">Searching for:</span>}
		<ul className="Tags" aria-label="Tags in your search">
			{chips.map(chip => renderChip(chip))}
			{offered.map(({ word, chips: group }) => [
				<li key={"for|" + word} className="TagFor">similar to “{word}”:</li>,
				...group.map(chip => renderChip(chip, "TagSuggested")),
			])}
		</ul>
		{notRecognised.length > 0 ? <span className="TagLabel">Not recognised: {notRecognised.join(", ")}.</span> : null}
	</div>;
}

// allowDescriptions shows the per-font description box; hidden for now, the endpoint is still there
export default function SearchPage({ username, onNeedLogin = () => {}, allowDescriptions = false }) {
	const initial = new URLSearchParams(window.location.search);
	const [text, setText] = useState(initial.get("q") || "");
	const [query, setQuery] = useState(initial.get("q") || "");
	const [page, setPage] = useState(Math.max(1, parseInt(initial.get("page"), 10) || 1));
	const [added, setAdded] = useState(listParam(initial, "tags"));      // tags added from suggestions ("-name" excludes)
	const [ignored, setIgnored] = useState(listParam(initial, "ignore")); // words whose guessed tags were removed
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
		if (added.length) params.set("tags", added.join(","));
		if (ignored.length) params.set("ignore", ignored.join(","));
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
	}, [query, page, added, ignored, username]);

	const navigate = useCallback((nextQuery, nextPage, nextAdded = [], nextIgnored = []) => {
		const params = new URLSearchParams();
		if (nextQuery) params.set("q", nextQuery);
		if (nextAdded.length) params.set("tags", nextAdded.join(","));
		if (nextIgnored.length) params.set("ignore", nextIgnored.join(","));
		if (nextPage > 1) params.set("page", nextPage);
		const search = params.toString();
		window.history.pushState({}, "", window.location.pathname + (search ? "?" + search : ""));
		setQuery(nextQuery);
		setPage(nextPage);
		setAdded(nextAdded);
		setIgnored(nextIgnored);
	}, []);

	// Back and forward buttons
	useEffect(() => {
		const onPop = () => {
			const params = new URLSearchParams(window.location.search);
			setText(params.get("q") || "");
			setQuery(params.get("q") || "");
			setPage(Math.max(1, parseInt(params.get("page"), 10) || 1));
			setAdded(listParam(params, "tags"));
			setIgnored(listParam(params, "ignore"));
		};
		window.addEventListener("popstate", onPop);
		return () => window.removeEventListener("popstate", onPop);
	}, []);

	// An empty box becomes a tick or a cross depending on where it came from: tick -> empty -> cross -> empty -> tick
	const nextAfterOff = useRef({});
	const setChoice = (chip, next) => {
		if (chip.state === "on") nextAfterOff.current[chip.name] = "neg";
		if (chip.state === "neg") nextAfterOff.current[chip.name] = "on";
		const others = added.filter(t => t.replace(/^-/, "") !== chip.name);
		const rest = ignored.filter(w => w !== chip.name);
		// changing the tags changes the results, so this goes back to page 1
		if (next === "on") navigate(query, 1, [...others, chip.name], rest);
		else if (next === "neg") navigate(query, 1, [...others, "-" + chip.name], rest);
		else navigate(query, 1, others, [...rest, chip.name]);
	};

	const goToPage = (p) => {
		navigate(query, p, added, ignored);
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
		<p style={{ fontSize: "3vmin", marginBottom: "4vh" }}>
			Please enter a description to search for a font
		</p>

		<form className="SearchForm" onSubmit={e => { e.preventDefault(); if (text.trim()) { nextAfterOff.current = {}; navigate(text.trim(), 1); } }}>
			<input type="text" name="description" aria-label="Describe a font" value={text}
				onChange={e => setText(e.target.value)} maxLength={200} />
		</form>

		<div ref={topRef} className="ResultsTop"></div>
		{error ? <p className="SearchMessage" role="alert">{error}</p> : null}

		{data === null && loading ? <p className="SearchMessage" role="status">Searching…</p> : null}
		{data === null ? null : <>
			<TagLine data={data} ignored={ignored} nextAfterOff={nextAfterOff} onChange={setChoice} />
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
