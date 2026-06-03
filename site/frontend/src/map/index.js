import './main.css'
import React, { useState, useRef, useEffect } from 'react';

export default function MapPage() {

    const [selectedMap, setSelectedMap] = useState("flower");

    const handleChange = (event) => {
        setSelectedMap(event.target.value);
    };

    return <div className="MapArea">
        <iframe title="mapLocation" src={"/" + selectedMap + ".html"} width="100%" height="100%" style={{ border: "none", minHeight: "100%" }}></iframe>
        <div>
            <label htmlFor="options">Font Map: </label>
            <select id="options" value={selectedMap} onChange={handleChange}>
                <option value="flower">Flower</option>
                <option value="blob">Blob</option>
            </select>
        </div>

    </div>
}