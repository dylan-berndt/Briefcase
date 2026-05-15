import './App.css';
import React, { useState, useRef } from 'react';
import { shaderMaterial } from '@react-three/drei';
import { extend, useFrame } from '@react-three/fiber';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';


const fragmentSource = `
#define PIXEL_SIZE 4.0f
#define CELL_SIZE 64

#define MOD 32

#define SPEED 0.75f

float interp(float a, float b, float t) {
    return (b - a) * t + a;
}

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 cellDir(ivec3 cell) {
    float z = rand(vec2(cell.x, cell.y + cell.z));
    float rxy = sqrt(1.0f - z * z);
    float phi = rand(vec2(cell.z, cell.x + cell.y));
    
    float y = rxy * sin(phi);
    float x = rxy * cos(phi);
    
    return normalize(vec3(x, y, z));
}

float perlin(vec3 position) {
    position = position / float(CELL_SIZE);
    ivec3 cellPos = ivec3(floor(position));
    
    int i = 0;
    
    float products[8];
    
    for (int z = 0; z <= 1; z++) {
        for (int y = 0; y <= 1; y++) {
            for (int x = 0; x <= 1; x++) {
                ivec3 checkCell = cellPos + ivec3(x, y, z);
                vec3 offset = position - vec3(checkCell) + vec3(0.01f);
                checkCell = checkCell % MOD;
                vec3 dir = cellDir(checkCell);
                
                float product = dot(dir, offset);
                products[i] = product;
                i += 1;
            }
        }
    }

    vec3 offset = position - vec3(cellPos);
    
    float xInterp[4] = float[4](
    interp(products[0], products[1], offset.x), 
    interp(products[2], products[3], offset.x), 
    interp(products[4], products[5], offset.x), 
    interp(products[6], products[7], offset.x)
    );
    float yInterp[2] = float[2](
    interp(xInterp[0], xInterp[1], offset.y), 
    interp(xInterp[2], xInterp[3], offset.y)
    );
    float zInterp = interp(yInterp[0], yInterp[1], offset.z);
    
    return zInterp > 0.0f ? zInterp * 2.0f + 0.5f : 0.0f;
}

float noise(vec3 position) {
    return perlin(position);
}

uniform float iTime;
varying vec2 vUv;
uniform vec2 resolution;

void main()
{
	vec2 v = vUv * resolution.x * 25.0f;
    vec2 uv = floor(v / PIXEL_SIZE) * PIXEL_SIZE;
    
    float time = (iTime + 16.0f) * SPEED * float(CELL_SIZE);

    float r = noise(vec3(uv * 1.4f, time));
    float g = noise(vec3(uv, -time - 2.0f));
    float b = noise(vec3(uv / 1.4f, time + 60.0f));
    
    vec3 col = vec3(r, g, b);
    
    //ivec3 cellPos = ivec3(vec3(uv, iTime * SPEED) / float(CELL_SIZE));
    //col = cellDir(cellPos);

    // Output to screen
    gl_FragColor = vec4(col * 1.4f,1.0);
}
`

const vertexSource = `
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const BackgroundMaterial = shaderMaterial(
	// Uniforms
	{ iTime: 0.0, resolution: new THREE.Vector2(1.0, 1.0) },
	// Vertex Shader
	vertexSource,
	// Fragment Shader
	fragmentSource
);

extend({ BackgroundMaterial });

function BackgroundShader({backgroundRef}) {
	const materialRef = useRef();

	useFrame((_, delta) => {
		if (materialRef.current) {
			materialRef.current.iTime += delta * 0.5;
		}
		if (backgroundRef.current) {
			materialRef.current.resolution.x = backgroundRef.current.offsetWidth;
			materialRef.current.resolution.y = backgroundRef.current.offsetHeight;
		}
	})

	return (
		<mesh scale={100}> {}
		<planeGeometry args={[1, 1]} /> {}
		<backgroundMaterial ref={materialRef} side={2} /> {}
		</mesh>
	)
}

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
		<div>
			<p>{message}</p>
			<form onSubmit={submitLogin}>
				<div style={{}}>
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
					<label htmlFor="password">Password:</label>
					<input
					type="password"
					id="password"
					name="password"
					value={loginForm.password}
					onChange={loginChange}
					/>
				</div>
				<button type="submit">Submit</button>
			</form>
		</div>
		<div>
			<button style={{border: registerToggle ? "transparent": "#888888 2px solid"}} onClick={() => {setRegisterToggle(false)}}>Login</button>
			<button style={{border: !registerToggle ? "transparent": "#888888 2px solid"}} onClick={() => {setRegisterToggle(true)}}>Register</button>
		</div>
	</div>
}


function App() {
	const [resultsFound, updateResultsFound] = useState(false);
	const [resultsIssue, setIssue] = useState("");
	const [results, setResults] = useState([]);

	const [query, setQuery] = useState("");

	const [loginVisible, setLoginVisible] = useState(false);

	const backgroundRef = useRef(null);

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
		<>
			<header className="Bar">
				<button className="LoginButton" onClick={() => {setLoginVisible(!loginVisible)}}>Login</button>
				{!loginVisible ? <></> : <LoginPopup></LoginPopup>}
			</header>
			<div className="App">
				<div className="Shader">
					<Canvas
					camera={{ position: [0, 0, 1] }} // Position the camera slightly back
					ref={backgroundRef}
					>
					<color attach="background" args={[0, 0, 0]} /> {/* Optional: Clear the scene color */}
					<BackgroundShader backgroundRef={backgroundRef}/>
					</Canvas>
				</div>
				<div className="Shadow">
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
									return <Result result={result} pangram={pangrams[index % pangrams.length]}></Result>
								})}
							</div>
						}
					</div>
				</div>
			</div>
		</>
	);
}

export default App;
