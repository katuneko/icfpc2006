"use strict";

const SUPPORTED = ["en", "ja", "zh-Hans", "de"];
const statusNode = document.querySelector("#load-status");

function chooseInitialLocale() {
  const query = new URLSearchParams(location.search).get("lang");
  if (SUPPORTED.includes(query)) return query;
  const stored = localStorage.getItem("afterimage-locale");
  if (SUPPORTED.includes(stored)) return stored;
  for (const language of navigator.languages || [navigator.language]) {
    if (language.toLowerCase().startsWith("ja")) return "ja";
    if (language.toLowerCase().startsWith("zh")) return "zh-Hans";
    if (language.toLowerCase().startsWith("de")) return "de";
  }
  return "en";
}

function renderFamilies(families) {
  const grid = document.querySelector("#family-grid");
  grid.replaceChildren(...families.map(([name, description, count], index) => {
    const article = document.createElement("article");
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    const heading = document.createElement("h3");
    heading.textContent = name;
    const body = document.createElement("p");
    body.textContent = description;
    const badge = document.createElement("b");
    badge.textContent = count;
    article.append(number, heading, body, badge);
    return article;
  }));
}

function renderFacts(facts) {
  const list = document.querySelector("#fact-list");
  list.replaceChildren(...facts.map(fact => {
    const item = document.createElement("li");
    item.textContent = fact;
    return item;
  }));
}

function applyLocale(locale, copy) {
  const selected = copy.locales[locale] || copy.locales.en;
  document.documentElement.lang = selected.htmlLang;
  document.documentElement.dataset.locale = locale;
  document.title = selected.editorialTitle;
  document.querySelectorAll("[data-copy]").forEach(node => {
    const key = node.dataset.copy;
    if (typeof selected[key] === "string") node.textContent = selected[key];
  });
  document.querySelector('meta[name="description"]').content = selected.lead;
  document.querySelector('meta[property="og:title"]').content = selected.editorialTitle;
  document.querySelector('meta[property="og:description"]').content = selected.headline;
  document.querySelector('meta[property="og:image"]').content = `../assets/social/afterimage-og-${locale}.png`;
  document.querySelectorAll("[data-locale-button]").forEach(button => {
    button.setAttribute("aria-pressed", button.dataset.localeButton === locale ? "true" : "false");
  });
  renderFamilies(selected.families);
  renderFacts(selected.facts);
  localStorage.setItem("afterimage-locale", locale);
  statusNode.textContent = `${selected.language} selected`;
}

async function main() {
  try {
    const response = await fetch("../copy/launch-copy.json", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const copy = await response.json();
    let locale = chooseInitialLocale();
    applyLocale(locale, copy);
    document.querySelectorAll("[data-locale-button]").forEach(button => {
      button.addEventListener("click", () => {
        locale = button.dataset.localeButton;
        applyLocale(locale, copy);
        const url = new URL(location.href);
        url.searchParams.set("lang", locale);
        history.replaceState({}, "", url);
      });
    });
  } catch (error) {
    statusNode.textContent = "English fallback loaded; localized copy requires an HTTP preview server.";
    console.error("Afterimage localization failed", error);
  }
}

main();
