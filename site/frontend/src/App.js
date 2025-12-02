import './App.css';
import React, { useState } from 'react';

function App() {
	const [token, setToken] = useState("");

	const [resultsFound, updateResultsFound] = useState(false);
	const [resultsIssue, setIssue] = useState("");
	const [results, setResults] = useState([]);

	const [query, setQuery] = useState("");

	const pangrams = [
		"The quick brown fox jumps over the lazy dog", 
		"A mad boxer shot a quick glove jab to the jaw of his dizzy opponent",
		"Mr. Jock asphyxiates; Numbered Dayz' fortune teller Vasquez was right",
		"Whenever the black fox jumped, the squirrel gazed suspiciously",
		"Jived zombies tackled the very quick fox",
		"Mr. Jock, TV quiz PhD, bags few lynx",
	]

	// TODO: Check that inputs are valid
	function getResults(enteredQuery) {
		fetch('/api/font/query?query=' + enteredQuery)
		.then(response => response.json())
		.then(json => {
			setResults(json.results)
			updateResultsFound(true);
		})
		.catch(error => setIssue(error))
	}

	async function loadFontFace(face) {
		const loadedFont = await face.load();
		document.fonts.add(loadedFont);
	}

	return (
		<div className="App">
			<div className="Shadow">
				<header className="Bar"></header>
				<div className="Center">
					<p style={{ fontSize: "6vmin", lineHeight: 1.8, textShadow: "black 0 10px 10px", marginTop: "-12vmin" }}>
						Font Search <br></br>
					</p>
					<div style={{ height: "6vmin" }}></div>
					<p style={{ fontSize: "3vmin" }}>
						Please enter a description to search for a font
					</p>
					{resultsIssue === "" ? <></> : <p>{resultsIssue.toString()}</p>}

					<input type="text" name="description" 
					style={{ fontSize: "3vmin", minWidth: "70vmin", minHeight: "4vmin" }}
					onChange={e => setQuery(e.target.value)}
					onKeyDown={e => {
						if (e.key === "Enter") getResults(query);
					}}></input>

					{!resultsFound ? <></> : 
						<div className="Results">
							{results.map((result, index) => {
								const face = FontFace(result.name, `url(${result.file})`);
								loadFontFace(face);
								return <a href={result.url}>
									<div>
										<p>{result.name} | Score: {result.score}</p>
										<p style={{"fontFamily": result.name}}>
											ABCDEFGHIJKLMNOPQRSTUVWXYZ<br></br>
											abcdefghijklmnopqrstuvwxyz<br></br>
											{pangrams[index % pangrams.length]}
										</p>
									</div>
								</a>
							})}
						</div>
					}
				</div>
			</div>

			
		</div>
	);
}

export default App;
