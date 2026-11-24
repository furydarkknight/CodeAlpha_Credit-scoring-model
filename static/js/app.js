// app.js - Client-side inference and dynamic UI updates

document.addEventListener("DOMContentLoaded", () => {
  // 1. Theme Management (Light / Dark)
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const htmlEl = document.documentElement;

  const currentTheme = localStorage.getItem("credit_app_theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

  function applyTheme(theme) {
    htmlEl.setAttribute("data-theme", theme);
    themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
    localStorage.setItem("credit_app_theme", theme);
  }

  applyTheme(currentTheme);

  themeToggle.addEventListener("click", () => {
    const nextTheme = htmlEl.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
  });

  // 2. Form & Controls
  const form = document.getElementById("scoringForm");
  const errorAlert = document.getElementById("errorAlert");
  const btnSubmit = document.getElementById("btnSubmit");
  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");

  // Result Elements
  const statusBadge = document.getElementById("statusBadge");
  const scoreValue = document.getElementById("scoreValue");
  const gaugeMeter = document.getElementById("gaugeMeter");
  const pdValue = document.getElementById("pdValue");
  const tierValue = document.getElementById("tierValue");
  const decisionBanner = document.getElementById("decisionBanner");
  const decisionText = document.getElementById("decisionText");
  const factorsList = document.getElementById("factorsList");

  // Preset Buttons
  const btnLowRisk = document.getElementById("btnLowRisk");
  const btnHighRisk = document.getElementById("btnHighRisk");

  const lowRiskPreset = {
    age: 45,
    MonthlyIncome: 8000,
    NumberOfDependents: 1,
    RevolvingUtilizationOfUnsecuredLines: 0.10,
    DebtRatio: 0.20,
    NumberOfOpenCreditLinesAndLoans: 8,
    NumberRealEstateLoansOrLines: 1,
    NumberOfTime30-59DaysPastDueNotWorse: 0,
    NumberOfTime60-89DaysPastDueNotWorse: 0,
    NumberOfTimes90DaysLate: 0
  };

  const highRiskPreset = {
    age: 30,
    MonthlyIncome: 3200,
    NumberOfDependents: 2,
    RevolvingUtilizationOfUnsecuredLines: 1.20,
    DebtRatio: 0.80,
    NumberOfOpenCreditLinesAndLoans: 12,
    NumberRealEstateLoansOrLines: 2,
    NumberOfTime30-59DaysPastDueNotWorse: 2,
    NumberOfTime60-89DaysPastDueNotWorse: 1,
    NumberOfTimes90DaysLate: 3
  };

  function populateForm(data) {
    for (const [key, val] of Object.entries(data)) {
      const input = document.getElementById(key);
      if (input) {
        input.value = val !== null ? val : "";
      }
    }
    errorAlert.classList.add("d-none");
  }

  btnLowRisk.addEventListener("click", () => populateForm(lowRiskPreset));
  btnHighRisk.addEventListener("click", () => populateForm(highRiskPreset));

  // 3. Score Gauge Animation (Arc circumference ~ 251.3)
  const TOTAL_ARC = 251.3;

  function updateGauge(score, tier) {
    // Clamp score between 300 and 850
    const clamped = Math.max(300, Math.min(850, score));
    const fraction = (clamped - 300) / (850 - 300);
    const offset = TOTAL_ARC * (1 - fraction);

    gaugeMeter.style.strokeDashoffset = offset;

    if (tier === "Low Risk") {
      gaugeMeter.style.stroke = "var(--success-color)";
    } else if (tier === "Moderate Risk") {
      gaugeMeter.style.stroke = "var(--warning-color)";
    } else {
      gaugeMeter.style.stroke = "var(--danger-color)";
    }
  }

  // 4. Submit & Prediction Call
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorAlert.classList.add("d-none");
    errorAlert.textContent = "";

    // Read form values
    const incomeRaw = document.getElementById("MonthlyIncome").value.trim();
    const dependentsRaw = document.getElementById("NumberOfDependents").value.trim();

    const payload = {
      age: parseInt(document.getElementById("age").value, 10),
      RevolvingUtilizationOfUnsecuredLines: parseFloat(document.getElementById("RevolvingUtilizationOfUnsecuredLines").value),
      DebtRatio: parseFloat(document.getElementById("DebtRatio").value),
      NumberOfOpenCreditLinesAndLoans: parseInt(document.getElementById("NumberOfOpenCreditLinesAndLoans").value, 10) || 0,
      NumberRealEstateLoansOrLines: parseInt(document.getElementById("NumberRealEstateLoansOrLines").value, 10) || 0,
      NumberOfTime30-59DaysPastDueNotWorse: parseInt(document.getElementById("NumberOfTime30-59DaysPastDueNotWorse").value, 10) || 0,
      NumberOfTime60-89DaysPastDueNotWorse: parseInt(document.getElementById("NumberOfTime60-89DaysPastDueNotWorse").value, 10) || 0,
      NumberOfTimes90DaysLate: parseInt(document.getElementById("NumberOfTimes90DaysLate").value, 10) || 0,
      MonthlyIncome: incomeRaw === "" ? null : parseFloat(incomeRaw),
      NumberOfDependents: dependentsRaw === "" ? null : parseFloat(dependentsRaw)
    };

    // UI Loading state
    btnSubmit.disabled = true;
    btnSpinner.classList.remove("d-none");
    btnText.textContent = "Calculating Default Risk...";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMsg = data.messages ? data.messages.join("<br>") : (data.error || "Prediction request failed.");
        errorAlert.innerHTML = errorMsg;
        errorAlert.classList.remove("d-none");
        return;
      }

      // Render Result Card
      scoreValue.textContent = data.credit_score;
      pdValue.textContent = data.probability_of_default_pct;
      tierValue.textContent = data.risk_tier;

      // Update Gauge
      updateGauge(data.credit_score, data.risk_tier);

      // Status Badge
      statusBadge.className = "status-badge";
      decisionBanner.className = "decision-banner";

      if (data.risk_tier === "Low Risk") {
        statusBadge.classList.add("badge-low-risk");
        statusBadge.textContent = "Low Risk (Approved)";
        decisionBanner.classList.add("banner-success");
      } else if (data.risk_tier === "Moderate Risk") {
        statusBadge.classList.add("badge-moderate-risk");
        statusBadge.textContent = "Moderate Risk (Conditional)";
        decisionBanner.classList.add("banner-warning");
      } else {
        statusBadge.classList.add("badge-high-risk");
        statusBadge.textContent = "High Risk (Declined)";
        decisionBanner.classList.add("banner-danger");
      }

      decisionText.textContent = data.decision;

      // Render Factors
      factorsList.innerHTML = "";
      if (data.factors && data.factors.length > 0) {
        data.factors.forEach((f) => {
          const li = document.createElement("li");
          li.textContent = f;
          factorsList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "None (Standard financial profile)";
        factorsList.appendChild(li);
      }

    } catch (err) {
      errorAlert.textContent = "Network error connecting to /predict endpoint: " + err.message;
      errorAlert.classList.remove("d-none");
    } finally {
      btnSubmit.disabled = false;
      btnSpinner.classList.add("d-none");
      btnText.textContent = "Evaluate Creditworthiness";
    }
  });

  // Automatically trigger low-risk evaluation on load so UI is populated
  populateForm(lowRiskPreset);
});
