const uploadBtn = document.getElementById('upload');
const fileInput = document.getElementById('file');
const result = document.getElementById('result');
const preview = document.getElementById('preview');


uploadBtn.addEventListener('click', async () => {
    const f = fileInput.files[0];
    if (!f) { alert('Choose an image'); return; }


// preview
preview.src = URL.createObjectURL(f);
preview.style.display = 'block';
document.querySelector('.preview-container').style.display = 'block';


const fd = new FormData();
fd.append('file', f);


result.innerHTML = '<span class="spinner" style="display:inline-block;">⏳</span> Classifying...';

try {
    const resp = await fetch(`${window.location.origin}/predict`, {
        method: 'POST',
        body: fd
    });
    const j = await resp.json();

    if (j.error) {
    result.textContent = 'Error: ' + j.error;
    } 
    else {
    result.innerHTML = `Cluster: <b>${j.cluster}</b> — Size: <b>${j.size}</b>`;
    }

    } catch (e) {
    result.textContent = 'Network error: ' + e.message;
    }
});