document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('pdf-upload');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    console.log(result);
});

document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const queryInput = document.getElementById('query').value;

    const response = await fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: queryInput, pdf_id: 1 })  // Use actual pdf_id
    });

    const result = await response.json();
    document.getElementById('response').innerText = result.response;
});
