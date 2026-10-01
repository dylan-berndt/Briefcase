import './App.css';
import React, { useState, useRef, useEffect } from 'react';
import { shaderMaterial } from '@react-three/drei';
import { extend, useFrame } from '@react-three/fiber';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import SearchPage from './search';
import AboutPage from './about';
import MapPage from './map';


const fragmentSource = `
#define PIXEL_SIZE 8.0f
#define CELL_SIZE 64

#define OCTAVES 3
#define LACUNARITY 2.0
#define GAIN 0.5

#define DIMMING 0.8

#define MOD 32

#define SPEED 0.6f

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
    float strength = 1.0;
    float zoom = 1.0;
    float total = 0.0;
    float weight = 0.0;
    
    for (int i = 0; i < OCTAVES; i++) {
        total += strength * perlin(position * zoom);
        weight += strength;
        
        strength *= GAIN;
        zoom *= LACUNARITY;
    }
    
    return total * DIMMING;
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

async function postForm(url, fields) {
	const response = await fetch(url, { body: new URLSearchParams(fields), method: "post" });
	const json = await response.json().catch(() => ({}));
	return { ok: response.ok, json };
}

function LoginPopup({ username, onAuth }) {
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

	const submitLogin = async (e) => {
		e.preventDefault();
		try {
			if (registerToggle) {
				// Register first and only then log in, the login needs the new account to exist
				const registered = await postForm('/api/font/register', loginForm);
				if (!registered.ok) {
					setMessage(registered.json.message || "Registration failed");
					return;
				}
			}

			const login = await postForm('/api/font/login', loginForm);
			setMessage(login.json.message || "");
			if (login.ok) {
				setLoginForm({ username: '', password: '' });
				onAuth(login.json.username);
			}
		} catch (error) {
			setMessage(error.message || String(error));
		}
	}

	const logout = async () => {
		await fetch('/api/font/logout', { method: "post" }).catch(() => {});
		onAuth(null);
	}

	if (username) {
		return <div className="LoginPopup">
			<p>Logged in as {username}</p>
			<button type="button" onClick={logout}>Log out</button>
		</div>
	}

	return <div className="LoginPopup">
		<div>
			<p role="status">{message}</p>
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
				<div style={{marginBottom: "2vh"}}>
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
	const [loginVisible, setLoginVisible] = useState(false);
	const [username, setUsername] = useState(null);

	useEffect(() => {
		fetch('/api/font/me')
			.then(response => response.json())
			.then(json => setUsername(json.username))
			.catch(() => {});
	}, []);

	const backgroundRef = useRef(null);

	const [location, setLocation] = useState("search");

	return (
		<>
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
					<header className="Bar">
						<div>
							<button className="HomeButton" onClick={() => setLocation("search")}>Home</button>
							<button className="HomeButton" onClick={() => setLocation("map")}>Map</button>
							<button className="HomeButton" onClick={() => setLocation("about")}>About</button>
						</div>
						<div>
							<button className="LoginButton" onClick={() => {setLoginVisible(!loginVisible)}}>{username || "Login"}</button>
							{!loginVisible ? <></> : <LoginPopup username={username} onAuth={(name) => {setUsername(name); setLoginVisible(false);}}></LoginPopup>}
						</div>
					</header>
					{location === "search" ? <SearchPage username={username} onNeedLogin={() => setLoginVisible(true)}></SearchPage> : (location === "about" ? <AboutPage></AboutPage> : <MapPage></MapPage>)}
				</div>
			</div>
		</>
	);
}

export default App;
