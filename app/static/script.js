const startStop = document.querySelector('.start-stop');
const stream = document.querySelector('.stream');
const webcam = document.querySelector('.webcam');
console.log(startStop);

startStop.addEventListener('click', () => {
  console.log('hi');
  startStop.classList.toggle('stop');
  startStop.classList.toggle('start');
  stream.classList.toggle('start');
  stream.classList.toggle('stop');
  webcam.classList.toggle('stop');
  if (startStop.classList.contains('stop')) {
    startStop.innerHTML = 'Stop';
  } else {
    startStop.innerHTML = 'Start';
  }
});
