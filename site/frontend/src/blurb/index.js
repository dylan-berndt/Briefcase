import './main.css'
import React, { useEffect, useState } from 'react';

// The short description of the site that closes the font pages and the search page. The words live on the server
// (aboutParagraphs in site/backend/seo.py), so a font page passes in the ones its own data came with and the search page
// asks for them.
export default function SiteBlurb({ paragraphs }) {
	const [fetched, setFetched] = useState(null);

	useEffect(() => {
		if (paragraphs) return;
		const controller = new AbortController();
		fetch('/api/font/about', { signal: controller.signal })
			.then(response => response.ok ? response.json() : Promise.reject())
			.then(json => setFetched(json.about))
			.catch(() => { });
		return () => controller.abort();
	}, [paragraphs]);

	const text = paragraphs || fetched;
	if (!text || !text.length) return null;
	return <section className="SiteBlurb">
		<h2>About Font Search</h2>
		{text.map((paragraph, i) => <p key={i}>{paragraph}</p>)}
	</section>;
}
