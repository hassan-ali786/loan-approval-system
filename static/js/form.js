const nextBtns = document.querySelectorAll(".next-btn");
const prevBtns = document.querySelectorAll(".prev-btn");
const formSteps = document.querySelectorAll(".form-step");
const progress = document.querySelector(".progress");
const stepNum = document.getElementById("stepNum");

let formStepNum = 0;

function updateProgressbar() {
    const pct = ((formStepNum + 1) / formSteps.length) * 100;
    progress.style.width = pct + "%";
    if (stepNum) stepNum.textContent = formStepNum + 1;
}

nextBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        formSteps[formStepNum].classList.remove("form-step-active");
        formStepNum++;
        formSteps[formStepNum].classList.add("form-step-active");
        updateProgressbar();
    });
});

prevBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        formSteps[formStepNum].classList.remove("form-step-active");
        formStepNum--;
        formSteps[formStepNum].classList.add("form-step-active");
        updateProgressbar();
    });
});

// Set initial progress on load
updateProgressbar();