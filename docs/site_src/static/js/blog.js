(function () {
  var themeButton = document.querySelector(".theme-toggle");
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("modelopt-theme", theme);
    if (themeButton) {
      var next = theme === "dark" ? "Light" : "Dark";
      var label = themeButton.querySelector(".theme-toggle-label");
      if (label) label.textContent = next;
      themeButton.setAttribute("aria-pressed", String(theme === "light"));
    }
  }
  setTheme(localStorage.getItem("modelopt-theme") || document.documentElement.dataset.theme || "dark");
  if (themeButton) {
    themeButton.addEventListener("click", function () {
      setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    });
  }

  var search = document.getElementById("post-search");
  var grid = document.getElementById("post-grid");
  var empty = document.getElementById("empty-state");
  var filter = document.getElementById("tag-filter");
  if (!search || !grid || !filter) return;

  var activeTag = "";

  function applyFilters() {
    var query = search.value.trim().toLowerCase();
    var visible = 0;
    grid.querySelectorAll(".post-card").forEach(function (card) {
      var haystack = [
        card.getAttribute("data-title") || "",
        card.getAttribute("data-summary") || "",
        card.getAttribute("data-tags") || "",
      ].join(" ");
      var tags = card.getAttribute("data-tags") || "";
      var matchesQuery = !query || haystack.indexOf(query) !== -1;
      var matchesTag = !activeTag || tags.split(/\s+/).indexOf(activeTag) !== -1;
      var show = matchesQuery && matchesTag;
      card.hidden = !show;
      if (show) visible += 1;
    });
    empty.hidden = visible !== 0;
  }

  search.addEventListener("input", applyFilters);
  filter.addEventListener("click", function (event) {
    var button = event.target.closest("button[data-tag]");
    if (!button) return;
    filter.querySelectorAll("button").forEach(function (item) {
      item.classList.toggle("active", item === button);
    });
    activeTag = button.getAttribute("data-tag") || "";
    applyFilters();
  });
})();
