var clockElement = document.getElementById('clock');

function clock() {
    clockElement.textContent = new Date().toString().slice(15, 24);
}

setInterval(clock, 1000);