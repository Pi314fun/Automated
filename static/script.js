document.addEventListener('DOMContentLoaded', () => {
  updateVisitorCount();

  // Connect to the socket
  const socket = io();
  socket.on('progress', handleProgress);

  const uploadForm = document.getElementById('upload-form');
  uploadForm.addEventListener('submit', () => {
    setTimeout(() => {
      emitFormSubmitted();
    }, 1000);
  });
});

function updateVisitorCount() {
  const visitorCount = localStorage.getItem('visitorCount') || 0;
  const newVisitorCount = parseInt(visitorCount) + 1;
  localStorage.setItem('visitorCount', newVisitorCount);
  document.getElementById('visitor-number').innerText = newVisitorCount;
}

function handleProgress(data) {
  const progressBar = document.getElementById('progress-bar');
  const progressMessage = document.getElementById('progress-message');
  const logsOutput = document.getElementById('logs-output');
  progressBar.style.width = `${data.progress}%`;
  progressMessage.innerText = data.message;

  // Add command line output to logs
  logsOutput.textContent += `${data.message}\n`;

  if (data.progress === 100) {
    window.location.href = "/completed";
  }
}

function emitFormSubmitted() {
  const useCase = document.querySelector('select[name="use_case"]').value;
  socket.emit('form_submitted', { use_case: useCase });
}
