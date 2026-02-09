const uploadBtn = document.getElementById('upload');
const fileInput = document.getElementById('file');
const result = document.getElementById('result');
const preview = document.getElementById('preview');


uploadBtn.addEventListener('click', async () => {
    const f = fileInput.files[0];
    if (!f) { alert('Choose an image'); return; }

    preview.src = URL.createObjectURL(f);
    preview.style.display = 'block';
    const previewContainer = document.querySelector('.preview-container');
    previewContainer.style.display = 'block';

    result.innerHTML = '<span class="spinner" style="display:inline-block;">⏳</span> Classifying...';

    const scanLine = document.querySelector('.scan-line');
    scanLine.style.display = 'block';
    scanLine.style.animation = 'scan 2s linear infinite';

    const fd = new FormData();
    fd.append('file', f);

    try {
        const resp = await fetch(`/predict`, {
            method: 'POST',
            body: fd
        });
        const j = await resp.json();

        scanLine.style.animation = 'none';

        if (j.error) {
            result.textContent = 'Error: ' + j.error;
        } else {
            result.innerHTML = `Cluster: <b>${j.cluster}</b> — Size: <b>${j.size}</b>`;
        }

    } catch (e) {
        scanLine.style.animation = 'none';
        result.textContent = 'Network error: ' + e.message;
    }
});
