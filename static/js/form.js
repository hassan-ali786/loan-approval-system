const form = document.getElementById("loanForm");
const nextBtns = document.querySelectorAll(".next-btn");
const prevBtns = document.querySelectorAll(".prev-btn");
const formSteps = document.querySelectorAll(".form-step");
const progress = document.querySelector(".progress");

let formStepNum = 0;

nextBtns.forEach(btn=>{
    btn.addEventListener("click", ()=>{
        formSteps[formStepNum].classList.remove("form-step-active");
        formStepNum++;
        formSteps[formStepNum].classList.add("form-step-active");
        updateProgressbar();
    });
});

prevBtns.forEach(btn=>{
    btn.addEventListener("click", ()=>{
        formSteps[formStepNum].classList.remove("form-step-active");
        formStepNum--;
        formSteps[formStepNum].classList.add("form-step-active");
        updateProgressbar();
    });
});

function updateProgressbar(){
    progress.style.width = ((formStepNum+1)/formSteps.length)*100 + "%";
}