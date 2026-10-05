import './main.css'
import React, { useState, useRef, useEffect } from 'react';

export default function MapPage() {

    const [selectedMap, setSelectedMap] = useState("flower");

    const handleChange = (event) => {
        setSelectedMap(event.target.value);
    };

    return <div className="MapArea">
        <div className="MapDescription">
            <h>Font Maps</h>
            <br></br><br></br>
            <p>This is Font Maps, a section of the site dedicated to visualizing the relationships between fonts. 
                Essentially, each font in the dataset is encoded into a vector that represents the visual information 
                present in that font. Fonts that are close to each other in this space typically look similar, and ones
                that are far apart will look different. We can squish the vectors down into 6 dimensions with tSNE and show
                them as positions in a 3D space, with colors defined by the last 3 of the 6 dimensions. So, if two fonts have
                a similar color, they also share some other features with each other.
                <br></br><br></br>
                I've generated a couple different maps made up of the different font datasets I've used and all the different models I
                trained before completing the search project. The flower map is made up of the embeddings extracted from one of the first
                models I trained that was designed to take a letter and uppercase it. It also only uses fonts from the Google Fonts
                repository. The next map is routes, and it contains every font that is available to search on the website embedded with the
                latest model that I created using LeVJEPA. </p>
            <br></br><br></br>
            <div height="4vh">
                <label htmlFor="options">Viewing: </label>
                <select id="options" value={selectedMap} onChange={handleChange}>
                    <option value="flower">Flower</option>
                    <option value="routes">Routes</option>
                </select>
            </div>
        </div>
        
        <iframe title="mapLocation" src={"/maps/" + selectedMap + ".html"} width="100%" height="90vh" style={{ border: "none"}}></iframe>
    </div>
}