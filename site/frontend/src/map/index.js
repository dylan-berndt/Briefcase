import './main.css'
import React, { useState, useRef, useEffect } from 'react';

export default function MapPage() {

    const [selectedMap, setSelectedMap] = useState("flower");

    const handleChange = (event) => {
        setSelectedMap(event.target.value);
    };

    return <div className="MapArea">
        <div height="4vh">
            <label htmlFor="options">Viewing: </label>
            <select id="options" value={selectedMap} onChange={handleChange}>
                <option value="flower">Flower</option>
                <option value="routes">Routes</option>
            </select>
        </div>
        <iframe title="mapLocation" src={"/maps/" + selectedMap + ".html"} width="100%" height="100% - 4vh" style={{ border: "none"}}></iframe>
    </div>
}