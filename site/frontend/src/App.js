import './App.css';
import React, { useState } from 'react';

function Result({result, pangram}) {
	const [showDescriptionField, setShowDescriptionField] = useState(false);
	const [description, setDescription] = useState("");
	const [message, setMessage] = useState("");

	async function loadFontFace(face) {
		const loadedFont = await face.load();
		document.fonts.add(loadedFont);
	}

	function submitDescription(enteredDescription) {
		fetch('/api/font/describe?description=' + enteredDescription)
		.then(response => response.json())
		.then(json => {
			setMessage(json.message)
		})
		.catch(error => setMessage(error))
	}

	const face = FontFace(result.name, `url(${result.file})`);
	loadFontFace(face);
	return <a href={result.url}>
		<div>
			<div>
				<p>{result.name} | Score: {result.score}</p>
				<p style={{"fontFamily": result.name}}>
					ABCDEFGHIJKLMNOPQRSTUVWXYZ<br></br>
					abcdefghijklmnopqrstuvwxyz<br></br>
					{pangram}
				</p>
			</div>
			<button onClick={() => {setShowDescriptionField(!showDescriptionField)}}></button>
			{!showDescriptionField ? <></> :
			<div className="DescriptionField">
				<p>Please provide a description for this font</p>
				<input type="text" name="description" 
					style={{ fontSize: "3vmin", minWidth: "70vmin", minHeight: "4vmin"}}
					onChange={e => setDescription(e.target.value)}
					onKeyDown={e => {
						if (e.key === "Enter") submitDescription(description);
					}}></input>
				<p>{message}</p>
			</div>}
		</div>
	</a>
}


function LoginPopup() {
	const [registerToggle, setRegisterToggle] = useState(false);
	const [loginForm, setLoginForm] = useState({
		username: '',
		password: ''
	})
	const [message, setMessage] = useState("");

	const loginChange = (e) => {
		setLoginForm({
			...loginForm,
			[e.target.name]: e.target.value
		})
	}

	const submitLogin = (e) => {
		e.preventDefault();
		if (registerToggle) {
			fetch('/api/font/register', {body: loginForm, method: "post"})
			.then(response => response.json())
			.then(data => {
				setMessage(data.message);
			})
			.catch(error => setMessage(error));
		}

		fetch('/api/font/login', {body: loginForm, method: "post"})
		.then(response => response.json())
		.then(data => {
			setMessage(data.message);
		})
		.catch(error => setMessage(error));
	}

	return <div className="LoginPopup">
		<p>{message}</p>
		<form onSubmit={submitLogin}>
			<div>
				<label htmlFor="username">Username:</label>
				<input
				type="text"
				id="username"
				name="username"
				value={loginForm.username}
				onChange={loginChange}
				/>
			</div>
			<div>
				<label htmlFor="email">Email:</label>
				<input
				type="email"
				id="email"
				name="email"
				value={loginForm.email}
				onChange={loginChange}
				/>
			</div>
			<button type="submit">Submit</button>
		</form>
		<div>
			<div>Login</div>
			<div>Register</div>
		</div>
	</div>
}


function App() {
	const [resultsFound, updateResultsFound] = useState(false);
	const [resultsIssue, setIssue] = useState("");
	const [results, setResults] = useState([]);

	const [query, setQuery] = useState("");

	const [loginVisible, setLoginVisible] = useState(false);

	const pangrams = [
		"The quick brown fox jumps over the lazy dog", 
		"A mad boxer shot a quick glove jab to the jaw of his dizzy opponent",
		"Mr. Jock asphyxiates; Numbered Dayz' fortune teller Vasquez was right",
		"Whenever the black fox jumped, the squirrel gazed suspiciously",
		"Jived zombies tackled the very quick fox",
		"Mr. Jock, TV quiz PhD, bags few lynx",
	]

	function getResults(enteredQuery) {
		fetch('/api/font/query?query=' + enteredQuery)
		.then(response => response.json())
		.then(json => {
			setResults(json.results)
			updateResultsFound(true);
		})
		.catch(error => setIssue(error))
	}

	return (
		<div className="App">
			<div className="Shadow">
				<div className="Center">
					<div className="Bar">
						<button className="LoginButton" onClick={() => {setLoginVisible(!loginVisible)}}>Login</button>
						{!loginVisible ? <></> : <LoginPopup></LoginPopup>}
					</div>
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
								return <Result result={result} pangram={pangrams[index % pangrams.length]}></Result>
							})}
						</div>
					}
				</div>
			</div>
		</div>
	);
}

export default App;
