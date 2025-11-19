import './App.css';
import React, { useState } from 'react';

function App() {
	const [resultsFound, updateResultsFound] = useState(false);
	const [resultsIssue, setIssue] = useState("");
	const [results, setResults] = useState([]);

	const [query, setQuery] = useState("");

	// TODO: Check that inputs are valid
	function getResults(enteredQuery) {
		fetch('/api/font/query/' + enteredQuery)
		.then(response => response.json())
		.then(json => {
			setResults(json)
			updateResultsFound(true);
		})
		.catch(error => setIssue(error))
	}

	return (
		<div className="App">
			<header className="App-header">
				<p style={{ fontSize: "6vmin", lineHeight: 1.8, textShadow: "black 0 10px 10px", marginTop: "-12vmin" }}>
					FontiFinder <br></br>
				</p>
				<div style={{ height: "6vmin" }}></div>
				<p style={{ fontSize: "3vmin" }}>
					Please enter a description to search for a font
				</p>
				{resultsIssue === "" ? <></> : <p>{resultsIssue}</p>}

				<input type="text" name="description" 
				style={{ fontSize: "3vmin", minWidth: "70vmin", minHeight: "4vmin" }}
				onChange={e => setQuery(e.target.value)}
				onKeyDown={e => {
					if (e.key === "Enter") getResults(query);
				}}></input>
			</header>

			{!resultsFound ? <></> : 
				<div className="Results">
					{results.map((result) => {
						return <p></p>
					})}
				</div>
			}
		</div>
	);
}

export default App;
