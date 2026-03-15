// ---------------------- Stars ----------------------
const canvas = document.getElementById('stars');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let stars = [];
for(let i=0;i<150;i++){
    stars.push({x: Math.random()*canvas.width, y: Math.random()*canvas.height, r: Math.random()*2, dx: (Math.random()-0.5)*0.2, dy: (Math.random()-0.5)*0.2});
}

function drawStars(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    stars.forEach(s=>{
        ctx.beginPath();
        ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
        ctx.fillStyle = 'white';
        ctx.fill();
        s.x += s.dx;
        s.y += s.dy;
        if(s.x<0) s.x=canvas.width;
        if(s.x>canvas.width) s.x=0;
        if(s.y<0) s.y=canvas.height;
        if(s.y>canvas.height) s.y=0;
    });
    requestAnimationFrame(drawStars);
}
drawStars();

// ---------------- Character Movement ----------------
const char = document.getElementById('character');
let charX = 0;
let charY = window.innerHeight - 100;
let direction = 1;

function moveChar(){
    charX += 1.5 * direction;
    if(charX > window.innerWidth - 64 || charX < 0) direction *= -1;
    char.style.left = charX + 'px';
    requestAnimationFrame(moveChar);
}
moveChar();

// ---------------- Character Click Explosion ----------------
const popup = document.getElementById('popup-panel');
char.addEventListener('click', ()=>{
    // Fire + paper explosion
    for(let i=0;i<30;i++){
        let p = document.createElement('div');
        p.style.position='fixed';
        p.style.width='8px';
        p.style.height='8px';
        p.style.borderRadius='50%';
        p.style.left = (charX + 16) + 'px';
        p.style.top = charY + 'px';
        p.style.background = (i%2===0)?'orange':'#fffacd';
        document.body.appendChild(p);
        let dx = (Math.random()-0.5)*300;
        let dy = (Math.random()-0.5)*300;
        let interval = setInterval(()=>{
            let left = parseFloat(p.style.left);
            let top = parseFloat(p.style.top);
            p.style.left = left + dx*0.02 + 'px';
            p.style.top = top + dy*0.02 + 'px';
        },16);
        setTimeout(()=>{clearInterval(interval); p.remove();}, 800);
    }
    popup.classList.add('show');
});

// Close popup
document.getElementById('close-popup').addEventListener('click', ()=>{
    popup.classList.remove('show');
});

// Algorithm info popup
const algoInfoPanel = document.getElementById('algo-info-panel');
const algoTitle = document.getElementById('algo-title');
const algoDescription = document.getElementById('algo-description');

document.querySelectorAll('#popup-panel .algo-buttons button').forEach(btn=>{
    btn.addEventListener('click', ()=>{
        const algo = btn.getAttribute('data-info');
        algoTitle.innerText = algo;
        if(algo === "Generator"){
            algoDescription.innerText = "Generator: CNN + ResBlocks + Attention for predicting ab channels.";
        } else if(algo === "Discriminator"){
            algoDescription.innerText = "Discriminator: CNN that judges real vs fake colorization images.";
        } else if(algo === "LPIPS"){
            algoDescription.innerText = "LPIPS: Learned perceptual image similarity using VGG network.";
        } else if(algo === "L1"){
            algoDescription.innerText = "L1 Loss: Pixel-wise difference between predicted and true ab channels.";
        }
        algoInfoPanel.classList.add('show');
    });
});

// Close Algorithm Info
document.getElementById('close-algo-info').addEventListener('click', ()=>{
    algoInfoPanel.classList.remove('show');
});

// ---------------- Drag & Drop Upload ----------------
const dropzone = document.getElementById('dropzone');
const fileInput = dropzone.querySelector('input[type=file]');
dropzone.addEventListener('click', ()=>fileInput.click());
dropzone.addEventListener('dragover', (e)=>{ e.preventDefault(); dropzone.classList.add('hover'); });
dropzone.addEventListener('dragleave', ()=>dropzone.classList.remove('hover'));
dropzone.addEventListener('drop', (e)=>{
    e.preventDefault();
    dropzone.classList.remove('hover');
    fileInput.files = e.dataTransfer.files;
});
