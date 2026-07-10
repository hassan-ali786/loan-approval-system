// ─── Multi-step form ─────────────────────────────
const nextBtns  = document.querySelectorAll(".next-btn");
const prevBtns  = document.querySelectorAll(".prev-btn");
const formSteps = document.querySelectorAll(".form-step");
const progress  = document.querySelector(".progress");
const stepNum   = document.getElementById("stepNum");

// Step dots
const dots = [
    document.getElementById('dot1'),
    document.getElementById('dot2'),
    document.getElementById('dot3'),
];

let currentStep = 0;

function updateUI() {
    const pct = ((currentStep + 1) / formSteps.length) * 100;
    if (progress) progress.style.width = pct + "%";
    if (stepNum) stepNum.textContent = currentStep + 1;

    // Update dots
    dots.forEach((dot, i) => {
        if (!dot) return;
        dot.classList.remove('active', 'done');
        if (i < currentStep)  dot.classList.add('done'), dot.textContent = '✓';
        if (i === currentStep) dot.classList.add('active'), dot.textContent = i + 1;
        if (i > currentStep)  dot.textContent = i + 1;
    });
}

nextBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        if (currentStep < formSteps.length - 1) {
            formSteps[currentStep].classList.remove("form-step-active");
            currentStep++;
            formSteps[currentStep].classList.add("form-step-active");
            updateUI();
        }
    });
});

prevBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        if (currentStep > 0) {
            formSteps[currentStep].classList.remove("form-step-active");
            currentStep--;
            formSteps[currentStep].classList.add("form-step-active");
            updateUI();
        }
    });
});

// Initialize
updateUI();