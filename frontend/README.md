# Frontend

This folder will host the Next.js frontend. For a quick start:

1. Install Node.js (16+)
2. From the repo root run:

npx create-next-app@latest frontend --use-npm --app

3. Create a simple page that calls the backend /query endpoint.

Sample fetch (browser):

fetch('/api/query', {method: 'POST', body: JSON.stringify({query: 'What is RAG?'})})
  .then(r => r.json())
  .then(console.log)

