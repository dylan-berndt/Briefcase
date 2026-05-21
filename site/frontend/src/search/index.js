import './main.css'
import React, { useState, useRef, useEffect } from 'react';

function Result({result, pangram, approveResult}) {
	const [showDescriptionField, setShowDescriptionField] = useState(false);
	const [description, setDescription] = useState("");
	const [message, setMessage] = useState("");

	const [approvedToggle, setApprovedToggle] = useState(false);

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
		<div className="ResultWindow">
			<div className="FontWindow">
				<div>
					<p>{result.name} | Score: {result.score}</p>
					<p style={{"fontFamily": result.name}}>
						ABCDEFGHIJKLMNOPQRSTUVWXYZ<br></br>
						abcdefghijklmnopqrstuvwxyz<br></br>
						{pangram}
					</p>
				</div>
				<button onClick={() => {setShowDescriptionField(!showDescriptionField)}}></button>
				<button onClick={() => {approveResult(result.name, setApprovedToggle, setMessage, approvedToggle)}}></button>
			</div>
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

export default function SearchPage() {
    const [resultsFound, updateResultsFound] = useState(false);
	const [resultsIssue, setIssue] = useState("");
	const [results, setResults] = useState([]);

	const [query, setQuery] = useState("");

    function getResults(enteredQuery) {
		fetch('/api/font/query?query=' + enteredQuery)
		.then(response => response.json())
		.then(json => {
			setResults(json.results)
			updateResultsFound(true);
		})
		.catch(error => setIssue(error))
	}

	function approveResult(fontName, setToggle, setResultError, revoke) {
		fetch('/api/font/approve?fontName=' + fontName + "&query=" + query + "&revoke=" + revoke.toString())
		.then(response => response.json())
		.then(json => {
			setToggle(!revoke);
		})
		.catch(error => setResultError(error.message || String(error)))
	}

	const fontClasses = ["pixelify", "aldrich", "google", "montserrat", "alfa", "montserrat", "google", "aldrich"];
	const [fontNum, setFontNum] = useState(0);

	useEffect(() => {
		const id = setInterval(() => {
			setFontNum(n => (n + 1) % fontClasses.length);
		}, 300);
		return () => clearInterval(id);
	}, []);
	  
	const displayFont = fontClasses[fontNum];

	const textRef = useRef(null);

	useEffect(() => {
		if (textRef.current) {
		  const el = textRef.current;
		  el.style.transform = "none";
		  const rendered = el.getBoundingClientRect().height;
		  const target = window.innerWidth * 0.08;
		  const scale = target / rendered;
		  el.style.transform = `scale(${scale})`;
		}
	  }, [displayFont]);

    const pangrams = [
		"The quick brown fox jumps over the lazy dog", 
		"A mad boxer shot a quick glove jab to the jaw of his dizzy opponent",
		"Mr. Jock asphyxiates; Numbered Dayz' fortune teller Vasquez was right",
		"Whenever the black fox jumped, the squirrel gazed suspiciously",
		"Jived zombies tackled the very quick fox",
		"Mr. Jock, TV quiz PhD, bags few lynx",
	]

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
                    return <Result result={result} pangram={pangrams[index % pangrams.length]} approveResult={approveResult}></Result>
                })}
            </div>
        }
    </div>
    )
}